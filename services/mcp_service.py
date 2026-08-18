"""MCP sonucu çözümleme yardımcıları ve istemci arayüzü."""

import json
from typing import Any, Dict, List, Optional


class MCPClient:
    """Sayaç servisinin kullandığı MCP istemcisi arayüzü."""

    def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError


def result_data(result: Dict[str, Any]) -> Dict[str, Any]:
    """MCP sonucundaki yapılandırılmış veriyi veya metin JSON'unu döndürür."""
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content", []):
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            parsed = json.loads(item.get("text", ""))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def tool_error_message(result: Dict[str, Any]) -> Optional[str]:
    """MCP hatasını kullanıcıya gösterilebilecek kısa bir metne dönüştürür."""
    if not result.get("isError", False):
        return None
    if result.get("error"):
        return str(result["error"])
    texts: List[str] = [
        str(item.get("text", ""))
        for item in result.get("content", [])
        if isinstance(item, dict) and item.get("type") == "text"
    ]
    return " ".join(text for text in texts if text) or "Bilinmeyen MCP araç hatası."
