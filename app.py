#!/usr/bin/env python3
"""Yerel Ollama sohbeti ve ThingsBoard MCP araç istemcisi."""

import json
import os
import re
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("APP_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
MAX_TOOL_ROUNDS = 2

SYSTEM_PROMPT = """/no_think
Sen ThingsBoard hava kalitesi sistemine bağlı Türkçe bir asistansın.
Kullanıcının sorusunu yanıtlamak için gerektiğinde MCP araçlarını kullan.
Cihaz veya ölçüm bilgilerini tahmin etme. Ölçüm istenirse gerekirse önce
cihazları listele, sonra dönen gerçek device_id ile telemetri aracını çağır.
Araç hatası oluşursa nedeni açıkça belirt. Cevapları kısa ve anlaşılır Türkçe
ile ver; sayısal ölçümlerde mümkün olduğunda birimleri belirt.
"""


def clean_model_answer(answer: str) -> str:
    """Model cevabına eklenebilen düşünme bölümünü kaldırır."""
    if "</think>" in answer:
        return answer.split("</think>", 1)[1].strip()
    return re.sub(r"^<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


class MCPClient:
    """Streamable HTTP kullanan küçük, bağımlılıksız MCP istemcisi."""

    def __init__(self, url: str):
        self.url = url
        self.session_id: Optional[str] = None
        self.request_id = 0

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    @staticmethod
    def _parse_response(raw: bytes, content_type: str) -> Dict[str, Any]:
        if not raw:
            return {}
        text = raw.decode("utf-8", errors="replace")
        if "text/event-stream" not in content_type:
            return json.loads(text)

        events: List[Dict[str, Any]] = []
        data_lines: List[str] = []
        for line in text.splitlines() + [""]:
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            elif not line and data_lines:
                events.append(json.loads("\n".join(data_lines)))
                data_lines = []
        if not events:
            raise ValueError("MCP sunucusu geçerli bir SSE verisi döndürmedi.")
        return events[-1]

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id

        request = Request(
            self.url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            new_session_id = response.headers.get("Mcp-Session-Id")
            if new_session_id:
                self.session_id = new_session_id
            return self._parse_response(
                response.read(), response.headers.get("Content-Type", "")
            )

    def connect(self) -> None:
        initialized = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "ollama-chat-app", "version": "1.0.0"},
            },
        })
        if "error" in initialized:
            raise RuntimeError(initialized["error"].get("message", "MCP başlatılamadı."))
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def list_tools(self) -> List[Dict[str, Any]]:
        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/list",
            "params": {},
        })
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "Araçlar alınamadı."))
        return response.get("result", {}).get("tools", [])

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        response = self._post({
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "tools/call",
            "params": {"name": name, "arguments": arguments},
        })
        if "error" in response:
            raise RuntimeError(response["error"].get("message", "Araç çalıştırılamadı."))
        return response.get("result", {})

    def close(self) -> None:
        if not self.session_id:
            return
        try:
            request = Request(
                self.url,
                headers={"Mcp-Session-Id": self.session_id},
                method="DELETE",
            )
            urlopen(request, timeout=10).close()
        except Exception:
            pass


def ollama_chat(
    model: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
) -> Dict[str, Any]:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.2,
            "num_predict": 256,
        },
    }).encode("utf-8")
    request = Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def mcp_tools_for_ollama(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": tool.get("inputSchema", {"type": "object", "properties": {}}),
        },
    } for tool in tools]


def run_agent(
    model: str,
    browser_messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    mcp = MCPClient(MCP_URL)
    events: List[Dict[str, Any]] = []
    try:
        mcp.connect()
        mcp_tools = mcp.list_tools()
        known_tools = {tool["name"] for tool in mcp_tools}
        tools = mcp_tools_for_ollama(mcp_tools)
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *browser_messages,
        ]
        completed_calls: Dict[str, Dict[str, Any]] = {}

        for _ in range(MAX_TOOL_ROUNDS):
            result = ollama_chat(model, messages, tools)
            assistant_message = result.get("message", {})
            messages.append(assistant_message)
            tool_calls = assistant_message.get("tool_calls") or []

            if not tool_calls:
                content = clean_model_answer(assistant_message.get("content", ""))
                return content or "Model bir yanıt üretemedi.", events

            for call in tool_calls:
                function = call.get("function", {})
                name = function.get("name", "")
                arguments = function.get("arguments") or {}
                if name not in known_tools:
                    tool_result: Dict[str, Any] = {
                        "isError": True,
                        "error": f"Bilinmeyen MCP aracı: {name}",
                    }
                elif not isinstance(arguments, dict):
                    tool_result = {
                        "isError": True,
                        "error": "Araç parametreleri JSON nesnesi olmalıdır.",
                    }
                else:
                    call_key = json.dumps(
                        {"name": name, "arguments": arguments},
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    )
                    if call_key in completed_calls:
                        tool_result = completed_calls[call_key]
                    else:
                        events.append({
                            "type": "tool_call",
                            "tool": name,
                            "arguments": arguments,
                        })
                        try:
                            tool_result = mcp.call_tool(name, arguments)
                        except Exception as exc:
                            tool_result = {"isError": True, "error": str(exc)}
                        completed_calls[call_key] = tool_result

                events.append({
                    "type": "tool_result",
                    "tool": name,
                    "ok": not bool(tool_result.get("isError", False)),
                    "result": tool_result,
                })
                messages.append({
                    "role": "tool",
                    "tool_name": name,
                    "content": json.dumps(tool_result, ensure_ascii=False, default=str),
                })

        messages.append({
            "role": "system",
            "content": (
                "Yeni araç çağırma. Yukarıdaki MCP sonuçlarını kullanarak "
                "kullanıcıya şimdi kısa ve doğrudan Türkçe yanıt ver."
            ),
        })
        final_result = ollama_chat(model, messages, [])
        final_content = clean_model_answer(
            final_result.get("message", {}).get("content", "")
        )
        return final_content or "Araç sonuçları alındı ancak model yanıt üretemedi.", events
    finally:
        mcp.close()


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(APP_DIR), **kwargs)

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            model = body.get("model", "qwen3:4b")
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": "En az bir mesaj gerekli."})
                return

            content, events = run_agent(model, messages)
            self._json(200, {"content": content, "events": events})
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._json(exc.code, {"error": f"Bağlantı hatası: {detail}"})
        except URLError as exc:
            self._json(503, {"error": f"Yerel servise ulaşılamadı: {exc.reason}"})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Geçersiz istek veya yanıt: {exc}"})
        except Exception as exc:
            self._json(500, {"error": f"Beklenmeyen hata: {exc}"})

    def _json(self, status: int, data: Dict[str, Any]) -> None:
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Ollama + ThingsBoard MCP Sohbeti: http://{HOST}:{PORT}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"MCP: {MCP_URL}")
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")
    finally:
        server.server_close()
