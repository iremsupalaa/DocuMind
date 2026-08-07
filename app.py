#!/usr/bin/env python3
"""Minimal local web chat for Ollama, using only Python's standard library."""

import json
import os
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("APP_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


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
            model = body.get("model", "gemma3:1b")
            messages = body.get("messages", [])

            if not isinstance(messages, list) or not messages:
                self._json(400, {"error": "En az bir mesaj gerekli."})
                return

            payload = json.dumps({
                "model": model,
                "messages": messages,
                "stream": False,
                "keep_alive":"30m",
            }).encode("utf-8")
            request = Request(
                f"{OLLAMA_URL}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=300) as response:
                result = json.loads(response.read())
            self._json(200, {"content": result["message"]["content"]})
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            self._json(exc.code, {"error": f"Ollama hatası: {detail}"})
        except URLError:
            self._json(503, {"error": "Ollama'ya ulaşılamadı. Ollama uygulamasının açık olduğundan emin olun."})
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._json(400, {"error": f"Geçersiz istek veya yanıt: {exc}"})
        except Exception as exc:
            self._json(500, {"error": f"Beklenmeyen hata: {exc}"})

    def _json(self, status, data):
        encoded = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Ollama Sohbet: http://{HOST}:{PORT}")
    print("Durdurmak için Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nSunucu durduruldu.")
    finally:
        server.server_close()
