#!/usr/bin/env python3
"""Yerel Ollama sohbeti ve ThingsBoard MCP araç istemcisi."""

import json
import hashlib
import os
import re
import sqlite3
import threading
import time
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
LIBRARY_DIR = Path(os.environ.get(
    "LIBRARY_DIR", str(Path.home() / "Desktop" / "Library-Connector")
)).expanduser()
LIBRARY_DB = Path(os.environ.get(
    "LIBRARY_DB", str(APP_DIR / "library.db")
)).expanduser()
LIBRARY_SCAN_SECONDS = float(os.environ.get("LIBRARY_SCAN_SECONDS", "2"))
MAX_TOOL_ROUNDS = 2

SYSTEM_PROMPT = """/no_think
Sen ThingsBoard hava kalitesi sistemine ve yerel belge kütüphanesine bağlı
Türkçe bir asistansın.
Kullanıcının sorusunu yanıtlamak için gerektiğinde MCP araçlarını kullan.
Cihaz veya ölçüm bilgilerini tahmin etme. Ölçüm istenirse gerekirse önce
cihazları listele, sonra dönen gerçek device_id ile telemetri aracını çağır.
Araç hatası oluşursa nedeni açıkça belirt. Cevapları kısa ve anlaşılır Türkçe
ile ver; sayısal ölçümlerde mümkün olduğunda birimleri belirt.
"""

LIBRARY_SYSTEM_PROMPT = """/no_think
Sen yerel bir belge kütüphanesini kullanan Türkçe bir asistansın.
Soruyu yalnızca aşağıdaki güncel belge parçalarına dayanarak cevapla.
Belgelerde bulunmayan bilgileri uydurma. Bilgi yetersizse bunu açıkça söyle.
Cevabı kısa, doğrudan ve Türkçe ver. İç planlamanı veya düşünme adımlarını yazma.
"""


def clean_model_answer(answer: str) -> str:
    """Model cevabına eklenebilen düşünme bölümünü kaldırır."""
    if "</think>" in answer:
        return answer.split("</think>", 1)[1].strip()
    return re.sub(r"^<think>.*?</think>", "", answer, flags=re.DOTALL).strip()


def result_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """MCP sonucundaki yapılandırılmış veriyi, yoksa metin JSON'unu döndürür."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content", []):
        if item.get("type") == "text":
            try:
                parsed = json.loads(item.get("text", ""))
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    return {}


def latest_value(telemetry: Dict[str, Any], key: str) -> Optional[str]:
    values = telemetry.get(key)
    if isinstance(values, list) and values:
        return str(values[0].get("value", "-"))
    if values is not None:
        return str(values)
    return None


def format_air_quality_result(
    devices: List[Dict[str, Any]],
    context: Optional[Dict[str, Any]] = None,
    measurements: Optional[Dict[str, Any]] = None,
) -> str:
    """Sık kullanılan hava kalitesi sorgularını modelsiz ve güvenilir biçimde biçimler."""
    if not devices:
        return "ThingsBoard hesabında hava kalitesi cihazı bulunamadı."

    lines = [f"ThingsBoard hesabında {len(devices)} cihaz bulundu:"]
    for index, device in enumerate(devices, 1):
        lines.append(
            f"{index}. {device.get('name', 'Adsız cihaz')} "
            f"(ID: {device.get('id', '-')}, tür: {device.get('type', '-')})"
        )

    first = devices[0]
    if context:
        attributes = context.get("attributes", context)
        if isinstance(attributes, list):
            attributes = {
                item.get("key"): item.get("value")
                for item in attributes
                if isinstance(item, dict) and item.get("key")
            }
        if isinstance(attributes, dict):
            details = []
            for key, label in (
                ("building", "bina"), ("floor", "kat"), ("room", "oda"),
                ("model", "model"), ("firmwareVersion", "firmware"),
            ):
                if attributes.get(key) is not None:
                    details.append(f"{label}: {attributes[key]}")
            if details:
                lines.append(f"İlk cihazın bilgileri — {', '.join(details)}.")

    if measurements:
        telemetry = measurements.get("telemetry", measurements)
        labels = (
            ("temperature", "sıcaklık", "°C"),
            ("humidity", "nem", "%"),
            ("co2", "CO₂", "ppm"),
            ("pm25", "PM2.5", "µg/m³"),
            ("pm10", "PM10", "µg/m³"),
            ("voc", "VOC", ""),
            ("aqi", "AQI", ""),
            ("battery", "batarya", "%"),
        )
        values = []
        for key, label, unit in labels:
            value = latest_value(telemetry, key)
            if value is not None:
                values.append(f"{label}: {value}{unit}")
        if values:
            lines.append(f"{first.get('name', 'İlk cihaz')} son ölçümleri — {', '.join(values)}.")
        aqi = latest_value(telemetry, "aqi")
        if aqi is not None:
            try:
                aqi_number = float(aqi)
                level = (
                    "iyi" if aqi_number <= 50 else
                    "orta" if aqi_number <= 100 else
                    "hassas gruplar için sağlıksız" if aqi_number <= 150 else
                    "sağlıksız"
                )
                lines.append(f"Kısa değerlendirme: AQI {aqi}, hava kalitesi {level} seviyededir.")
            except ValueError:
                pass
    return "\n\n".join(lines)


def tool_error_message(result: Dict[str, Any]) -> Optional[str]:
    """MCP araç hatasını kullanıcıya gösterilebilecek kısa metne dönüştürür."""
    if not result.get("isError", False):
        return None
    if result.get("error"):
        return str(result["error"])
    texts = [
        str(item.get("text", ""))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return " ".join(filter(None, texts)) or "Bilinmeyen MCP araç hatası."


def looks_like_internal_planning(answer: str) -> bool:
    """Modelin son cevap yerine İngilizce iç planlama üretip üretmediğini saptar."""
    lowered = answer.casefold()
    markers = (
        "okay, the user", "the user wants me", "first, i need",
        "wait, the user", "let me check the tools", "i should call",
    )
    return sum(marker in lowered for marker in markers) >= 2


class LibraryIndex:
    """Yerel metin dosyalarını SQLite ile sürekli eşitleyen küçük belge indeksi."""

    SUPPORTED_SUFFIXES = {".txt", ".md"}
    STOP_WORDS = {
        "acaba", "ama", "bana", "ben", "bir", "bu", "da", "de", "diye",
        "en", "gibi", "hakkında", "hangi", "ile", "için", "kim", "mi",
        "mı", "mu", "mü", "nasıl", "ne", "nedir", "neden", "olan", "olarak",
        "onu", "şu", "ve", "veya",
    }

    def __init__(self, folder: Path, database: Path):
        self.folder = folder
        self.database = database
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_sync_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.folder.mkdir(parents=True, exist_ok=True)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize_database(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS documents (
                    path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL REFERENCES documents(path) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    UNIQUE(path, chunk_index)
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path);
                PRAGMA optimize;
            """)

    @staticmethod
    def _chunks(text: str, target_size: int = 1200, overlap: int = 180) -> List[str]:
        normalized = re.sub(r"\r\n?", "\n", text).strip()
        if not normalized:
            return []
        chunks: List[str] = []
        start = 0
        while start < len(normalized):
            end = min(start + target_size, len(normalized))
            if end < len(normalized):
                boundary = max(
                    normalized.rfind("\n", start + target_size // 2, end),
                    normalized.rfind(". ", start + target_size // 2, end),
                )
                if boundary > start:
                    end = boundary + 1
            part = normalized[start:end].strip()
            if part:
                chunks.append(part)
            if end >= len(normalized):
                break
            start = max(end - overlap, start + 1)
        return chunks

    def sync(self) -> Dict[str, int]:
        """Eklenen/değişen belgeleri indeksler, silinenleri veritabanından kaldırır."""
        with self._lock:
            discovered: Dict[str, Path] = {}
            for file_path in self.folder.rglob("*"):
                if file_path.is_file() and file_path.suffix.casefold() in self.SUPPORTED_SUFFIXES:
                    discovered[str(file_path.relative_to(self.folder))] = file_path

            added = changed = deleted = 0
            with self._connect() as connection:
                stored = {
                    row["path"]: row
                    for row in connection.execute(
                        "SELECT path, mtime_ns, size, sha256 FROM documents"
                    )
                }
                for relative_path, file_path in discovered.items():
                    stat = file_path.stat()
                    previous = stored.get(relative_path)
                    if (
                        previous
                        and previous["mtime_ns"] == stat.st_mtime_ns
                        and previous["size"] == stat.st_size
                    ):
                        continue

                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if previous and previous["sha256"] == digest:
                        connection.execute(
                            "UPDATE documents SET mtime_ns = ?, size = ? WHERE path = ?",
                            (stat.st_mtime_ns, stat.st_size, relative_path),
                        )
                        continue

                    connection.execute("DELETE FROM documents WHERE path = ?", (relative_path,))
                    connection.execute(
                        "INSERT INTO documents(path, mtime_ns, size, sha256, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (relative_path, stat.st_mtime_ns, stat.st_size, digest, time.time()),
                    )
                    connection.executemany(
                        "INSERT INTO chunks(path, chunk_index, content) VALUES (?, ?, ?)",
                        [
                            (relative_path, index, chunk)
                            for index, chunk in enumerate(self._chunks(content))
                        ],
                    )
                    if previous:
                        changed += 1
                    else:
                        added += 1

                for relative_path in set(stored) - set(discovered):
                    connection.execute("DELETE FROM documents WHERE path = ?", (relative_path,))
                    deleted += 1

            self.last_sync_at = time.time()
            self.last_error = None
            return {"added": added, "changed": changed, "deleted": deleted}

    @staticmethod
    def _tokens(text: str) -> List[str]:
        return [
            token for token in re.findall(r"[\wçğıöşüÇĞİÖŞÜ]{3,}", text.casefold())
            if token not in LibraryIndex.STOP_WORDS
        ]

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Soru ile ortak anlamlı kelimeleri bulunan en ilgili belge parçalarını seçer."""
        self.sync()
        tokens = self._tokens(query)
        if not tokens:
            return []
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT path, chunk_index, content FROM chunks"
            ).fetchall()

        scored: List[Tuple[int, sqlite3.Row]] = []
        for row in rows:
            haystack = f"{row['path']} {row['content']}".casefold()
            score = sum((3 if token in row["path"].casefold() else 1) * haystack.count(token) for token in tokens)
            if score:
                scored.append((score, row))
        scored.sort(key=lambda item: (-item[0], item[1]["path"], item[1]["chunk_index"]))
        return [
            {
                "path": row["path"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "score": score,
            }
            for score, row in scored[:limit]
        ]

    def status(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunks = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return {
            "folder": str(self.folder),
            "documents": documents,
            "chunks": chunks,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
        }

    def watch(self) -> None:
        while not self._stop.is_set():
            try:
                self.sync()
            except Exception as exc:
                self.last_error = str(exc)
            self._stop.wait(LIBRARY_SCAN_SECONDS)

    def start(self) -> None:
        threading.Thread(target=self.watch, name="library-connector", daemon=True).start()


library_index = LibraryIndex(LIBRARY_DIR, LIBRARY_DB)


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
    events: List[Dict[str, Any]] = []
    user_query = next((
        str(message.get("content", ""))
        for message in reversed(browser_messages)
        if message.get("role") == "user"
    ), "")
    normalized_query = user_query.casefold()
    is_thingsboard_request = (
        "thingsboard" in normalized_query
        or "list_air_quality_devices" in normalized_query
        or "get_latest_air_quality" in normalized_query
        or "get_device_context" in normalized_query
    ) and any(word in normalized_query for word in ("cihaz", "hava", "ölçüm", "aqi"))

    if not is_thingsboard_request:
        library_hits = library_index.search(user_query)
        if library_hits:
            context = "\n\n".join(
                f"[Kaynak: {hit['path']} | Parça: {hit['chunk_index'] + 1}]\n"
                f"{hit['content']}"
                for hit in library_hits
            )[:10000]
            events.append({
                "type": "library_retrieval",
                "tool": "Library Connector",
                "ok": True,
                "result": {
                    "sources": sorted({hit["path"] for hit in library_hits}),
                    "chunk_count": len(library_hits),
                },
            })
            result = ollama_chat(model, [
                {"role": "system", "content": LIBRARY_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Güncel belge parçaları:\n\n{context}\n\nSoru: {user_query}\n/no_think",
                },
            ], [])
            answer = clean_model_answer(result.get("message", {}).get("content", ""))
            if looks_like_internal_planning(answer):
                return "Model belge cevabı yerine iç planlama üretti. Soruyu daha kısa yazın.", events
            return answer or "Belgeler bulundu ancak model yanıt üretemedi.", events

        result = ollama_chat(model, [
            {
                "role": "system",
                "content": (
                    "/no_think\nKısa, anlaşılır ve Türkçe cevap ver. "
                    "Kullanıcı yerel kütüphanedeki bir bilgiyi soruyor ancak "
                    "eşleşen belge bulunamadıysa bunu açıkça belirt; bilgi uydurma."
                ),
            },
            *browser_messages,
        ], [])
        answer = clean_model_answer(result.get("message", {}).get("content", ""))
        return answer or "Model bir yanıt üretemedi.", events

    mcp = MCPClient(MCP_URL)
    try:
        mcp.connect()
        mcp_tools = mcp.list_tools()
        known_tools = {tool["name"] for tool in mcp_tools}
        tools = mcp_tools_for_ollama(mcp_tools)

        if is_thingsboard_request:
            def call_and_record(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                events.append({"type": "tool_call", "tool": name, "arguments": arguments})
                try:
                    result = mcp.call_tool(name, arguments)
                except Exception as exc:
                    result = {"isError": True, "error": str(exc)}
                events.append({
                    "type": "tool_result",
                    "tool": name,
                    "ok": not bool(result.get("isError", False)),
                    "result": result,
                })
                return result

            list_result = call_and_record(
                "list_air_quality_devices", {"page_size": 10, "search": ""}
            )
            error = tool_error_message(list_result)
            if error:
                return f"ThingsBoard cihazları alınamadı: {error}", events

            devices_payload = result_data(list_result)
            devices = devices_payload.get("devices", [])
            if not isinstance(devices, list):
                devices = []

            wants_context = any(word in normalized_query for word in (
                "konum", "model", "firmware", "bina", "oda", "kat",
                "get_device_context",
            ))
            wants_measurements = any(word in normalized_query for word in (
                "ölçüm", "sıcaklık", "sicaklik", "nem", "co₂", "co2",
                "pm2.5", "pm10", "voc", "aqi", "batarya", "değerlendir",
                "get_latest_air_quality",
            ))

            context_payload: Optional[Dict[str, Any]] = None
            measurements_payload: Optional[Dict[str, Any]] = None
            if devices and (wants_context or wants_measurements):
                device_id = str(devices[0].get("id", ""))
                if not device_id:
                    return "İlk cihazın kimliği ThingsBoard yanıtında bulunamadı.", events

                if wants_context:
                    context_result = call_and_record(
                        "get_device_context", {"device_id": device_id}
                    )
                    error = tool_error_message(context_result)
                    if error:
                        return f"Cihaz bilgileri alınamadı: {error}", events
                    context_payload = result_data(context_result)

                if wants_measurements:
                    measurements_result = call_and_record(
                        "get_latest_air_quality", {"device_id": device_id}
                    )
                    error = tool_error_message(measurements_result)
                    if error:
                        return f"Son hava kalitesi ölçümleri alınamadı: {error}", events
                    measurements_payload = result_data(measurements_result)

            return format_air_quality_result(
                devices, context_payload, measurements_payload
            ), events

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
                if looks_like_internal_planning(content):
                    return (
                        "Model araç çağrısına geçemedi. İsteği daha kısa biçimde "
                        "yeniden gönderin.", events
                    )
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

    def do_GET(self):
        if self.path == "/api/library/status":
            try:
                sync_result = library_index.sync()
                status = library_index.status()
                status["sync"] = sync_result
                self._json(200, status)
            except Exception as exc:
                self._json(500, {"error": f"Kütüphane eşitlenemedi: {exc}"})
            return

        super().do_GET()

    def do_POST(self):
        if self.path != "/api/chat":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            model = body.get("model", "gemma3:4b")
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
    first_sync = library_index.sync()
    library_index.start()
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Ollama + Library Connector Sohbeti: http://{HOST}:{PORT}")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"MCP: {MCP_URL}")
    print(f"Kütüphane: {LIBRARY_DIR}")
    print(
        "İlk eşitleme: "
        f"{first_sync['added']} eklendi, "
        f"{first_sync['changed']} güncellendi, "
        f"{first_sync['deleted']} silindi"
    )
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")
    finally:
        server.server_close()
