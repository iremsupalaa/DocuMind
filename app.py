#!/usr/bin/env python3
"""Yerel Ollama sohbeti ve ThingsBoard MCP araç istemcisi."""

import json
import hashlib
import os
import re
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_wtf.csrf import CSRFProtect

from auth import (
    authenticate_user,
    get_user_by_id,
    get_user_by_thingsboard_email,
)
from services.library_service import (
    LibraryIndex,
    get_library_members,
    load_library_indexes,
    resolve_library_access,
    search_libraries,
    set_library_indexes,
)
from services.meter_service import answer_meter_question

from repositories.user_repository import (
    ensure_thingsboard_identity_schema,
)

from services.thingsboard_auth_service import (
    ThingsBoardAuthError,
    authenticate_thingsboard_user,
)

APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("APP_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
LIBRARY_DIR = Path(os.environ.get(
    "LIBRARY_DIR", str(Path.home() / "Desktop" / "Library-Connector")
)).expanduser()
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql:///ollama_library"
)
LIBRARY_SCAN_SECONDS = float(os.environ.get("LIBRARY_SCAN_SECONDS", "2"))
MAX_TOOL_ROUNDS = 2
METER_MODEL_FORMATTING = os.environ.get(
    "METER_MODEL_FORMATTING", "true"
).strip().casefold() in {"1", "true", "yes", "on"}

SYSTEM_PROMPT = """/no_think
Sen ThingsBoard hava kalitesi sistemine ve yerel belge kütüphanesine bağlı
Türkçe bir asistansın.
Kullanıcının sorusunu yanıtlamak için gerektiğinde MCP araçlarını kullan.
Cihaz veya ölçüm bilgilerini tahmin etme. Ölçüm istenirse gerekirse önce
cihazları listele, sonra dönen gerçek device_id ile telemetri aracını çağır.
Araç hatası oluşursa nedeni açıkça belirt. Cevapları kısa ve anlaşılır Türkçe
ile ver; sayısal ölçümlerde mümkün olduğunda birimleri belirt.
"""
METER_TOOL_NAMES = {
    "list_meter_devices",
    "get_latest_meter_reading",
    "get_meter_context",
    "get_meter_history",
    "analyze_meter_data_quality",
    "analyze_meter_metric",
    "get_meter_daily_energy_series",
    "rank_meters_by_interval_energy",
    "get_meter_energy_summary",
    "get_meter_connection_status",
    "compare_meter_devices",
    "rank_meters_by_metric",
    "get_meter_fleet_summary",
    "find_meter_anomalies",
    "group_meters_by_attribute",
}

METER_SYSTEM_PROMPT = """
/no_think
Sen NuSafe akıllı sayaç asistanısın.

Yalnızca sana sağlanan MCP araç sonuçlarına dayanarak cevap ver.
Sayaç değerlerini, tarihleri veya tüketim bilgilerini uydurma.
Kullanıcı bir sayaç belirtmemişse önce list_meter_devices aracını kullan.
Enerji tüketimi için get_meter_energy_summary,
anlık değerler için get_latest_meter_reading,
geçmiş için get_meter_history,
null veya eksik telemetri analizi için analyze_meter_data_quality,
min, maks, ortalama ve eşik analizi için analyze_meter_metric,
günlük tüketim serisi için get_meter_daily_energy_series,
zaman aralığı tüketim sıralaması için rank_meters_by_interval_energy,
bağlantı için get_meter_connection_status aracını kullan.
Birden fazla sayacı karşılaştırmak için compare_meter_devices,
sıralama için rank_meters_by_metric,
genel özet için get_meter_fleet_summary,
anomali taraması için find_meter_anomalies,
attribute gruplaması için group_meters_by_attribute aracını kullan.
Kısa, anlaşılır ve Türkçe yanıt ver.
"""

LIBRARY_SYSTEM_PROMPT = """/no_think
Sen yerel bir belge kütüphanesini kullanan Türkçe bir asistansın.
Soruyu yalnızca aşağıdaki güncel belge parçalarına dayanarak cevapla.
Belgelerde açıkça yazmayan hiçbir bilgiyi tahmin etme veya varsayma.
Bilgi yetersizse yalnızca "Bu bilgi belgelerde belirtilmiyor." diye cevapla.
Cevabı kısa, doğrudan ve Türkçe ver. İç planlamanı veya düşünme adımlarını yazma.
"""

EMBEDDING_MODEL = os.environ.get(
    "EMBEDDING_MODEL",
    "embeddinggemma"
)
EMBEDDING_DIM = int(os.environ.get("EMBEDDING_DIM", "768"))

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

SECRET_KEY = os.environ.get("SECRET_KEY")

if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY tanımlanmamış. "
        "Uygulamayı çalıştırmadan önce export SECRET_KEY=... kullanın."
    )


app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
    static_folder=str(APP_DIR / "static"),
)

app.config.update(
    SECRET_KEY=SECRET_KEY,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
    SESSION_COOKIE_SECURE=False,
)

ensure_thingsboard_identity_schema()

login_manager = LoginManager()
login_manager.login_view = "login_page"
login_manager.init_app(app)

csrf = CSRFProtect(app)

@login_manager.user_loader
def load_user(user_id):
    return get_user_by_id(user_id)


@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/"):
        return jsonify({
            "error": "Bu işlem için giriş yapmalısınız."
        }), 401

    return redirect(url_for("login_page"))


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
    include_device_list: bool = True,
) -> str:
    """Sık kullanılan hava kalitesi sorgularını modelsiz ve güvenilir biçimde biçimler."""
    if not devices:
        return "ThingsBoard hesabında hava kalitesi cihazı bulunamadı."

    lines: List[str] = []
    if include_device_list:
        lines.append(f"ThingsBoard hesabında {len(devices)} hava kalitesi cihazı bulundu:")
        for index, device in enumerate(devices, 1):
            lines.append(
                f"{index}. {device.get('name', 'Adsız cihaz')} "
                f"(ID: {device.get('id', '-')}, tür: {device.get('type', '-')})"
            )

    first = devices[0]
    if context:
        # MCP sunucuları bağlam verisini doğrudan sözlük, ``attributes``
        # altında sözlük ya da anahtar/değer listesi olarak döndürebilir.
        # Hepsini ortak bir sözlüğe dönüştürürüz.
        attributes = context.get("attributes")
        if attributes is None:
            attributes = context.get("context")
        if attributes is None:
            attributes = context.get("data")
        if attributes is None:
            attributes = context
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
                lines.append(
                    f"{first.get('name', 'Cihaz')} cihaz bilgileri — "
                    f"{', '.join(details)}."
                )
            else:
                lines.append(
                    f"{first.get('name', 'Cihaz')} için bina, kat, oda, model "
                    "veya firmware bilgisi ThingsBoard'da tanımlanmamış."
                )

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


def is_air_quality_device(device: Dict[str, Any]) -> bool:
    """Eski genel cihaz listesinden yalnızca hava kalitesi cihazlarını ayıklar."""
    device_type = str(device.get("type", "")).strip().casefold()
    return device_type in {
        "default",
        "air_quality",
        "air-quality",
        "air quality",
    }


def thingsboard_error_message(action: str, error: str) -> str:
    """Teknik MCP/HTTP ayrıntılarını arayüz yerine sunucu günlüğünde tutar."""
    print(f"[ThingsBoard hata] {action}: {error}")
    return (
        f"{action} şu anda ThingsBoard'dan alınamadı. "
        "Bağlantıyı kontrol edip kısa süre sonra tekrar deneyin."
    )


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






library_indexes = load_library_indexes()
set_library_indexes(library_indexes)



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


def installed_ollama_models() -> List[str]:
    """Yerel Ollama'da kurulu model etiketlerini döndürür."""
    request = Request(
        f"{OLLAMA_URL}/api/tags",
        headers={"Accept": "application/json"},
        method="GET",
    )
    with urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())

    return [
        str(item.get("name", "")).strip()
        for item in payload.get("models", [])
        if str(item.get("name", "")).strip()
    ]


def validate_ollama_model(model: Any) -> str:
    """Arayüzden gelen model adını temizler ve kurulu olduğunu doğrular."""
    selected = str(model or DEFAULT_OLLAMA_MODEL).strip()
    if not selected:
        selected = DEFAULT_OLLAMA_MODEL
    if not re.fullmatch(r"[A-Za-z0-9._/-]+(?::[A-Za-z0-9._-]+)?", selected):
        raise ValueError("Geçersiz Ollama model adı.")

    installed = installed_ollama_models()
    if selected not in installed:
        available = ", ".join(installed) or "kurulu model yok"
        raise ValueError(
            f"'{selected}' Ollama'da kurulu değil. Kurulu modeller: {available}"
        )
    return selected


def format_meter_answer_with_model(
    model: str,
    answer: str,
) -> Tuple[str, bool]:
    """Kısa sayaç sonucunu modele yazdırır; uzun listeleri aynen korur."""
    # Bir dil modeline yüzlerce doğrulanmış satırı yeniden yazdırmak hem
    # yavaştır hem de satır atlama/uydurma riski taşır. Liste ve uzun sonuçlar
    # meter_service tarafından üretildiği haliyle eksiksiz döndürülür.
    if len(answer) > 1800 or answer.count("\n") > 35:
        return answer, False
    protected_messages = (
        "belirtin",
        "en az iki",
        "bulunamadı",
        "alınamadı",
        "yapılamadı",
        "tanımlanmamış",
        "hata:",
        "enerji karşılaştırması:",
        "faz gerilim karşılaştırması:",
        "faz akım karşılaştırması:",
        "faz güç faktörü karşılaştırması:",
        "ölçümüne göre sayaç sıralaması:",
        "kez veri göndermiştir",
    )
    if any(message in answer.casefold() for message in protected_messages):
        return answer, False

    result = ollama_chat(
        model,
        [
            {
                "role": "system",
                "content": (
                    "/no_think\nYalnızca Türkçe cevap ver. Aşağıdaki doğrulanmış "
                    "sayaç sonucunu doğrudan ve kısa biçimde sun. Giriş cümlesi, "
                    "yorum, açıklama veya başlık ekleme. ESM gibi kısaltmaları "
                    "açıklamaya çalışma. Hiçbir sayı, birim, sayaç adı, tarih, "
                    "sıralama veya durumu değiştirme; yeni bilgi ekleme."
                ),
            },
            {
                "role": "user",
                "content": f"Doğrulanmış araç sonucu:\n\n{answer}",
            },
        ],
        [],
    )
    formatted = clean_model_answer(
        result.get("message", {}).get("content", "")
    )
    return (formatted or answer), True


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
    mode: str = "auto",
    user_libraries: Optional[
        List[Tuple[str, str, LibraryIndex]]
    ] = None,
    can_use_meter: bool = False,
    tb_customer_id: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    user_query = next((
        str(message.get("content", ""))
        for message in reversed(browser_messages)
        if message.get("role") == "user"
    ), "")
    normalized_query = user_query.casefold()
    air_quality_terms = (
        "hava kalitesi", "hava kalite", "sıcaklık", "sicaklik", "nem",
        "co₂", "co2", "pm2.5", "pm10", "voc", "aqi", "batarya",
        "list_air_quality_devices", "get_latest_air_quality",
        "hava kalitesi cihazı", "hava kalite cihazı",
    )
    generic_device_terms = (
        "thingsboard", "cihaz", "cihazlar", "telemetri", "ölçüm",
    )
    meter_terms = (
        "sayaç", "sayac", "meter", "kwh", "kilovat", "kilowatt",
        "enerji tüketimi", "toplam enerji", "anlık güç",
        "aktif güç", "gerilim", "voltaj", "akım", "amper",
        "frekans", "güç faktörü", "power factor",
        "bağlantı durumu", "çevrim içi", "çevrimdışı",
        "list_meter_devices", "get_latest_meter_reading",
        "get_meter_context", "get_meter_history",
        "get_meter_energy_summary",
        "get_meter_connection_status",
    )
    has_meter_identifier = bool(re.search(
        r"\b(?:esm-\d+|4001001_\d+)\b",
        normalized_query,
        flags=re.IGNORECASE,
    ))
    has_air_quality_intent = any(
        term in normalized_query for term in air_quality_terms
    )
    has_generic_device_intent = any(
        term in normalized_query for term in generic_device_terms
    )
    has_meter_intent = (
        has_meter_identifier
        or any(term in normalized_query for term in meter_terms)
        or (
            can_use_meter
            and has_generic_device_intent
            and not has_air_quality_intent
        )
    )

    is_meter_request = ( #sayaç isteğinin gerçekten devreye alınacağı kararı verilir (sayaç yetkisi + soruda sayaç niyeti)
        can_use_meter
        and has_meter_intent
    )
    
    has_thingsboard_intent = has_air_quality_intent
    # Otomatik modda hava kalitesiyle ilgili doğal dil sorularını,
    # belge aramasından önce ThingsBoard akışına yönlendir. Sayaç akışı
    # daha özel olduğu için önceliklidir.
    is_thingsboard_request = (
        has_thingsboard_intent
        and not is_meter_request
    )

    if mode == "meter": #kullanıcı arayüzden sayaç modunu seçtiğinde soru çok kısa bile olsa sayaç akışı zorunlu hale gelir 
        if not can_use_meter:
            return (
                "Bu kullanıcının sayaç ajanına erişim izni yok.",
                events,
            )
       
        is_meter_request = True
        is_thingsboard_request = False

    elif mode == "thingsboard":
        if not has_thingsboard_intent:
            return (
                "ThingsBoard modu yalnızca cihaz, sensör ve telemetri "
                "sorularını yanıtlar. Kütüphane soruları için çalışma "
                "modunu 'Kütüphane' olarak değiştirin.",
                events,
            )
        is_thingsboard_request = True
        is_meter_request = False

    elif mode in {"library", "general"}:
        is_thingsboard_request = False
        is_meter_request = False

    if mode == "general":
        result = ollama_chat(model, [
            {
                "role": "system",
                "content": (
                    "/no_think\nKısa, doğru ve anlaşılır Türkçe cevap ver. "
                    "Bilmediğin bilgileri uydurma."
                ),
            },
            *browser_messages,
        ], [])
        answer = clean_model_answer(result.get("message", {}).get("content", ""))
        return answer or "Model bir yanıt üretemedi.", events

    if not is_thingsboard_request and not is_meter_request:
        if not user_libraries:
          return (
        "Bu kullanıcı için erişilebilir bir kütüphane bulunmuyor.",
        events,
    )

        library_hits = search_libraries(
        user_query,
        user_libraries,
)
        if library_hits:
            context = "\n\n".join(
                f"[Sahip: {hit['owner_name']} | "f"Kaynak: {hit['path']} | "f"Parça: {hit['chunk_index'] + 1}]\n"
                f"{hit['content']}"
                for hit in library_hits
            )[:10000]
            events.append({
                "type": "library_retrieval",
                "tool": "Library Connector",
                "ok": True,
                "result": {
                    "sources": [
                        {
                            "path": hit["path"],
                            "chunk_index": hit["chunk_index"],
                            "score": round(hit["score"], 4),
                            "semantic_score": round(hit["semantic_score"], 4),
                            "keyword_score": round(hit["keyword_score"], 4),
                            "content": hit["content"],
                            "owner_id": hit["owner_id"],
                            "owner_name": hit["owner_name"],
                        }
                        for hit in library_hits
                    ],
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

        if mode == "library":
            return "Bu bilgi güncel kütüphane belgelerinde bulunamadı.", events

        result = ollama_chat(model, [
            {
                "role": "system",
                "content": (
                    "/no_think\nKısa, anlaşılır ve Türkçe cevap ver. "
                    "Kullanıcı yerel kütüphanedeki bir bilgiyi soruyor ancak "
                    "eşleşen belge bulunamadıysa yalnızca 'Bu bilgi belgelerde "
                    "bulunamadı.' de. Tahmin, varsayım veya uydurma bilgi ekleme."
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
        if is_meter_request:
            mcp_tools = [
                tool
                for tool in mcp_tools
                if tool["name"] in METER_TOOL_NAMES
            ]

        known_tools = {
            tool["name"]
            for tool in mcp_tools
        }
        tools = mcp_tools_for_ollama(mcp_tools)

        if is_thingsboard_request:
            def call_and_record(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
                print(f"[DEBUG] MCP çağrısı: {name} -> {arguments}")
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
                "list_air_quality_devices",
                {
                    "page_size": 10,
                    "search": "",
                    # boşsa tüm tenant cihazları,
                    # doluysa yalnızca o customer'a
                    # ThingsBoard tarafında atanmış cihazlar listelenir.
                    "customer_id": tb_customer_id,
                },
            )
            error = tool_error_message(list_result)
            if error:
                return thingsboard_error_message(
                    "Hava kalitesi cihazları", error
                ), events

            devices_payload = result_data(list_result)
            devices = devices_payload.get("devices", [])
            if not isinstance(devices, list):
                devices = []

            
            devices = [
                device
                for device in devices
                if isinstance(device, dict) and is_air_quality_device(device)
            ]

            is_listing_request = any(word in normalized_query for word in (
                "liste", "list", "hangi cihaz", "cihazlar",
            ))

            wants_context = any(word in normalized_query for word in (
                "konum", "model", "firmware", "bina", "oda", "kat",
                "seri numarası", "seri no", "bilgi", "detay",
                "özellik", "get_device_context",
            ))
            wants_measurements = any(word in normalized_query for word in (
                "ölçüm", "sıcaklık", "sicaklik", "nem", "co₂", "co2",
                "pm2.5", "pm10", "voc", "aqi", "batarya", "değerlendir",
                "telemetri", "son veri", "get_latest_air_quality",
            ))

           
            named_devices = [
                device for device in devices
                if str(device.get("name", "")).casefold() in normalized_query
            ]
            selected_devices = named_devices or devices
            if (
                not is_listing_request
                and (wants_context or wants_measurements)
                and len(selected_devices) > 1
            ):
                names = ", ".join(
                    str(device.get("name", "Adsız cihaz"))
                    for device in selected_devices
                )
                return (
                    f"Hangi hava kalitesi cihazını kastediyorsunuz? "
                    f"Şunlardan birini yazın: {names}.",
                    events,
                )

            context_payload: Optional[Dict[str, Any]] = None
            measurements_payload: Optional[Dict[str, Any]] = None
            if selected_devices and (wants_context or wants_measurements):
                device_id = str(selected_devices[0].get("id", ""))
                if not device_id:
                    return "Seçilen cihazın kimliği ThingsBoard yanıtında bulunamadı.", events

                if wants_context:
                    context_result = call_and_record(
                        "get_device_context", {"device_id": device_id}
                    )
                    error = tool_error_message(context_result)
                    if error:
                        return thingsboard_error_message(
                            "Cihaz bilgileri", error
                        ), events
                    context_payload = result_data(context_result)

                if wants_measurements:
                    measurements_result = call_and_record(
                        "get_latest_air_quality", {"device_id": device_id}
                    )
                    error = tool_error_message(measurements_result)
                    if error:
                        return thingsboard_error_message(
                            "Son hava kalitesi ölçümleri", error
                        ), events
                    measurements_payload = result_data(measurements_result)

            return format_air_quality_result(
                selected_devices,
                context_payload,
                measurements_payload,
                include_device_list=is_listing_request,
            ), events

        if is_meter_request:
            meter_answer, meter_events = answer_meter_question(
                query=user_query,
                mcp=mcp,
            )
            events.extend(meter_events)

            deterministic_tools = {
                "analyze_meter_data_quality",
                "analyze_meter_metric",
                "get_meter_daily_energy_series",
                "rank_meters_by_interval_energy",
                "get_meter_history",
                "get_meter_energy_summary",
                "compare_meter_devices",
                "rank_meters_by_metric",
                "get_meter_fleet_summary",
                "find_meter_anomalies",
                "group_meters_by_attribute",
            }
            used_tools = {
                str(event.get("tool", ""))
                for event in meter_events
                if event.get("type") == "tool_call"
            }
            if METER_MODEL_FORMATTING and not (used_tools & deterministic_tools):
                started_at = time.perf_counter()
                meter_answer, model_used = format_meter_answer_with_model(
                    model,
                    meter_answer,
                )
                events.append({
                    "type": (
                        "model_response"
                        if model_used
                        else "model_skipped"
                    ),
                    "model": model,
                    "elapsed_seconds": round(
                        time.perf_counter() - started_at,
                        3,
                    ),
                    "reason": (
                        None
                        if model_used
                        else "Uzun/liste sonucu doğruluğu korumak için aynen döndürüldü."
                    ),
                })

            return meter_answer, events


        messages: List[Dict[str, Any]] = [
                 {
                "role": "system",
                "content": (
                    METER_SYSTEM_PROMPT
                    if is_meter_request
                    else SYSTEM_PROMPT
                ),
            },
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


@app.route("/login", methods=["GET", "POST"])

def login_page():
    """ThingsBoard hesabıyla giriş yapılmasını sağlar."""

    if current_user.is_authenticated:
        return redirect(url_for("chat_page"))

    error = None
    email = ""

    if request.method == "POST":
        # "username" eski login.html ile geçici uyumluluk sağlar.
        email = (
            request.form.get("email")
            or request.form.get("username", "")
        ).strip().lower()

        password = request.form.get("password", "")

        if not email or not password:
            error = "E-posta adresi ve parola zorunludur."
        else:
            try:
                # E-posta ve parola ThingsBoard üzerinden doğrulanır.
                tb_session = authenticate_thingsboard_user(
                    email=email,
                    password=password,
                )

                # ThingsBoard hesabı yerel DocuMind kullanıcısıyla eşleştirilir.
                user = get_user_by_thingsboard_email(
                    tb_session.user.email
                )

                if user is None:
                    error = (
                        "ThingsBoard girişi başarılı, ancak bu hesap "
                        "DocuMind kullanıcısıyla eşleştirilmemiş."
                    )
                elif not user.is_active:
                    error = "DocuMind hesabınız aktif değil."
                else:
                   
                    session["tb_customer_id"] = ( #login basarılıysa customer_id oturuma yazılıyor
                        tb_session.user.customer_id or ""
                    )
                    login_user(user)
                    return redirect(url_for("chat_page"))

            except ThingsBoardAuthError as exception:
                
                print(f"[ThingsBoard giriş hatası] e-posta={email}: {exception}")
                error = str(exception)

    return render_template(
        "login.html",
        error=error,
        email=email,
    )


@app.post("/logout")
@login_required
def logout():
    session.pop("tb_customer_id", None) #logoutta siliniyor 
    logout_user()
    return redirect(url_for("login_page"))


@app.get("/")
@login_required
def chat_page():
    members = get_library_members()

    library_scopes = [
        {
            "id": user_id,
            "display_name": display_name,
        }
        for user_id, display_name in members.items()
    ]

    return render_template(
        "chat.html",
        current_user=current_user,
        library_scopes=library_scopes,
    )


@app.get("/api/library/status")
@login_required
def library_status():
    try:
        requested_scope = request.args.get(
            "scope",
            "self",
        )

        selected_libraries = resolve_library_access(
            current_user,
            requested_scope,
        )

        document_count = 0
        chunk_count = 0
        library_details = []

        for owner_id, owner_name, library_index in selected_libraries:
            sync_result = library_index.sync()
            status = library_index.status()

            document_count += status["documents"]
            chunk_count += status["chunks"]

            library_details.append({
                "owner_id": owner_id,
                "owner_name": owner_name,
                "folder": status["folder"],
                "documents": status["documents"],
                "chunks": status["chunks"],
                "sync": sync_result,
            })

        return jsonify({
            "documents": document_count,
            "chunks": chunk_count,
            "scope": requested_scope,
            "libraries": library_details,
        })

    except ValueError as exc:
        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:
        return jsonify({
            "error": (
                "Kütüphane durumu alınamadı: "
                f"{exc}"
            )
        }), 500


@app.post("/api/chat")
@login_required
def api_chat():
    try:
        body = request.get_json(silent=True) or {}

        model = validate_ollama_model(
            body.get("model", DEFAULT_OLLAMA_MODEL)
        )
        mode = body.get("mode", "auto")
        messages = body.get("messages", [])
        requested_scope = str( body.get("library_scope", "self"))

        if mode not in {
            "auto",
            "library",
            "general",
            "thingsboard",
            "meter",
        }:
            
            mode = "auto"
        if mode == "meter" and not current_user.can_use_meter: #sayaç ajanına erişim kontrolu
            return jsonify({
                "error": (
                    "Sayaç ajanına erişim yetkiniz bulunmuyor."
                )
            }), 403

        if not isinstance(messages, list) or not messages:
            return jsonify({
                "error": "En az bir mesaj gerekli."
            }), 400

        selected_libraries = resolve_library_access(
            current_user,
            requested_scope,
)

        content, events = run_agent(
            model,
            messages,
            mode,
            user_libraries=selected_libraries,
            can_use_meter=current_user.can_use_meter,
            tb_customer_id=session.get("tb_customer_id", ""),
        )

        return jsonify({
            "content": content,
            "events": events,
            "model": model,
            "model_used": bool(
                mode != "meter" or METER_MODEL_FORMATTING
            ),
        })

    except HTTPError as exc:
        detail = exc.read().decode(
            "utf-8",
            errors="replace",
        )

        return jsonify({
            "error": f"Bağlantı hatası: {detail}"
        }), exc.code

    except URLError as exc:
        return jsonify({
            "error": (
                "Yerel servise ulaşılamadı: "
                f"{exc.reason}"
            )
        }), 503

    except (ValueError, KeyError) as exc:
        return jsonify({
            "error": f"Geçersiz istek veya yanıt: {exc}"
        }), 400

    except Exception as exc:
        return jsonify({
            "error": f"Beklenmeyen hata: {exc}"
        }), 500


if __name__ == "__main__":
    first_syncs: Dict[str, Dict[str, int]] = {}
    for user_id, user_library in library_indexes.items():
        first_syncs[user_id] = user_library.sync()
        user_library.start()

    print(
        f"Ollama + Library Connector Sohbeti: "
        f"http://{HOST}:{PORT}"
    )
    print(f"Giriş sayfası: http://{HOST}:{PORT}/login")
    print(f"Ollama: {OLLAMA_URL}")
    print(f"Varsayılan model: {DEFAULT_OLLAMA_MODEL}")
    print(
        "Sayaç cevaplarında model biçimlendirmesi: "
        f"{'açık' if METER_MODEL_FORMATTING else 'kapalı'}"
    )
    print(f"MCP: {MCP_URL}")
    print(f"Kullanıcı kütüphanesi sayısı: {len(library_indexes)}")
    print("Veritabanı: PostgreSQL + pgvector")
    print(
        f"Embedding: {EMBEDDING_MODEL} "
        f"({EMBEDDING_DIM} boyut)"
    )
    for user_id, sync_result in first_syncs.items():
        user_library = library_indexes[user_id]
        print(
            f"Kullanıcı {user_id} ({user_library.folder}) ilk eşitleme: "
            f"{sync_result['added']} eklendi, "
            f"{sync_result['changed']} güncellendi, "
            f"{sync_result['deleted']} silindi"
        )
    print("Durdurmak için Ctrl+C")

    app.run(
        host=HOST,
        port=PORT,
        threaded=True,
        use_reloader=False,
    )
