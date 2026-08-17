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
)

APP_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
PORT = int(os.environ.get("APP_PORT", "8080"))
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MCP_URL = os.environ.get("MCP_URL", "http://127.0.0.1:8000/mcp")
LIBRARY_DIR = Path(os.environ.get(
    "LIBRARY_DIR", str(Path.home() / "Desktop" / "Library-Connector")
)).expanduser()
DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql:///ollama_library"
)
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
METER_TOOL_NAMES = {
    "list_meter_devices",
    "get_latest_meter_reading",
    "get_meter_context",
    "get_meter_history",
    "get_meter_energy_summary",
    "get_meter_connection_status",
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
bağlantı için get_meter_connection_status aracını kullan.
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

def ollama_embeddings(metinler):
    if not metinler:
        return []

    payload = json.dumps({
        "model": EMBEDDING_MODEL,
        "input": metinler,
        "dimensions": EMBEDDING_DIM,
        "keep_alive": "30m"
    }).encode("utf-8")

    request = Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urlopen(request, timeout=300) as response:
        result = json.loads(response.read())

    embeddings = result.get("embeddings", [])

    if len(embeddings) != len(metinler):
        raise ValueError(
            "Embedding sayısı metin sayısıyla eşleşmedi."
        )

    return embeddings


class LibraryIndex:
    """Yerel metin dosyalarını PostgreSQL + pgvector ile sürekli eşitler."""

    SUPPORTED_SUFFIXES = {".txt", ".md"}
    STOP_WORDS = {
        "acaba", "ama", "bana", "ben", "bir", "bu", "da", "de", "diye",
        "en", "gibi", "hakkında", "hangi", "ile", "için", "kim", "mi",
        "mı", "mu", "mü", "nasıl", "ne", "nedir", "neden", "olan", "olarak",
        "onu", "şu", "ve", "veya",
    }

    def __init__(self, owner_id: int, folder: Path, database_url: str):
        self.owner_id = int(owner_id)
        self.folder = folder
        self.database_url = database_url
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.last_sync_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.folder.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _connect(self) -> psycopg.Connection:
        connection = psycopg.connect(
            self.database_url,
            row_factory=dict_row,
        )
        register_vector(connection)
        return connection

    def _initialize_database(self) -> None:
        with psycopg.connect(
            self.database_url,
            autocommit=True,
            row_factory=dict_row,
        ) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(connection)

            connection.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id BIGSERIAL PRIMARY KEY,
                    owner_id BIGINT NOT NULL
                        REFERENCES users(id) ON DELETE CASCADE,
                    path TEXT NOT NULL,
                    mtime_ns BIGINT NOT NULL,
                    size BIGINT NOT NULL,
                    sha256 TEXT NOT NULL,
                    embedding_model TEXT NOT NULL,
                    embedding_dim INTEGER NOT NULL,
                    updated_at DOUBLE PRECISION NOT NULL,
                    UNIQUE(owner_id, path)
                )
            """)
            connection.execute(f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id BIGSERIAL PRIMARY KEY,
                    document_id BIGINT NOT NULL
                        REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({EMBEDDING_DIM}) NOT NULL,
                    UNIQUE(document_id, chunk_index)
                )
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_documents_owner
                ON documents(owner_id)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_document
                ON chunks(document_id)
            """)
            connection.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine
                ON chunks USING hnsw (embedding vector_cosine_ops)
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
        """Bu kullanıcıya ait eklenen/değişen belgeleri indeksler, silinenleri kaldırır."""
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
                        "SELECT id, path, mtime_ns, size, sha256, "
                        "embedding_model, embedding_dim FROM documents "
                        "WHERE owner_id = %s",
                        (self.owner_id,),
                    )
                }
                for relative_path, file_path in discovered.items():
                    stat = file_path.stat()
                    previous = stored.get(relative_path)
                    content = file_path.read_text(encoding="utf-8", errors="replace")
                    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
                    if (
                        previous
                        and previous["sha256"] == digest
                        and previous["embedding_model"] == EMBEDDING_MODEL
                        and previous["embedding_dim"] == EMBEDDING_DIM
                    ):
                        if (
                            previous["mtime_ns"] != stat.st_mtime_ns
                            or previous["size"] != stat.st_size
                        ):
                            connection.execute(
                                "UPDATE documents SET mtime_ns = %s, size = %s "
                                "WHERE id = %s AND owner_id = %s",
                                (
                                    stat.st_mtime_ns,
                                    stat.st_size,
                                    previous["id"],
                                    self.owner_id,
                                ),
                            )
                        continue

                    chunks = self._chunks(content)
                    embeddings = ollama_embeddings(chunks)
                    if len(chunks) != len(embeddings):
                        raise ValueError(
                            "Parça ve embedding sayıları eşleşmedi."
                        )
                    if any(len(embedding) != EMBEDDING_DIM for embedding in embeddings):
                        raise ValueError(
                            f"Embedding boyutu {EMBEDDING_DIM} olmalıdır."
                        )

                    if previous:
                        connection.execute(
                            "DELETE FROM documents "
                            "WHERE id = %s AND owner_id = %s",
                            (previous["id"], self.owner_id),
                        )

                    document = connection.execute(
                        "INSERT INTO documents("
                        "owner_id, path, mtime_ns, size, sha256, embedding_model, "
                        "embedding_dim, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                        "RETURNING id",
                        (
                            self.owner_id,
                            relative_path,
                            stat.st_mtime_ns,
                            stat.st_size,
                            digest,
                            EMBEDDING_MODEL,
                            EMBEDDING_DIM,
                            time.time(),
                        ),
                    ).fetchone()
                    document_id = document["id"]

                    if chunks:
                        with connection.cursor() as cursor:
                            cursor.executemany(
                                "INSERT INTO chunks("
                                "document_id, chunk_index, content, embedding) "
                                "VALUES (%s, %s, %s, %s)",
                                [
                                    (
                                        document_id,
                                        index,
                                        chunk,
                                        Vector(embedding),
                                    )
                                    for index, (chunk, embedding) in enumerate(
                                        zip(chunks, embeddings)
                                    )
                                ],
                            )
                    if previous:
                        changed += 1
                    else:
                        added += 1

                for relative_path in set(stored) - set(discovered):
                    connection.execute(
                        "DELETE FROM documents "
                        "WHERE id = %s AND owner_id = %s",
                        (stored[relative_path]["id"], self.owner_id),
                    )
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
        #pgvector cosine araması en ilgili belge parçalarını pgvector veritabanından bulur
        self.sync()
        query_embedding = ollama_embeddings([query])[0]
        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Sorgu embedding boyutu {EMBEDDING_DIM} olmalıdır."
            )
        query_vector = Vector(query_embedding)
        query_tokens = self._tokens(query)
        candidate_limit = max(limit * 10, 50)

        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    d.path,
                    c.chunk_index,
                    c.content,
                    GREATEST(
                        0.0,
                        1.0 - (c.embedding <=> %s)
                    ) AS semantic_score
                FROM chunks AS c
                JOIN documents AS d ON d.id = c.document_id
                WHERE d.owner_id = %s
                ORDER BY c.embedding <=> %s
                LIMIT %s
                """,
                (
                    query_vector,
                    self.owner_id,
                    query_vector,
                    candidate_limit,
                ),
            ).fetchall()

        if not rows:
            return []

        results: List[Dict[str, Any]] = []
        highest_keyword_score = 1

        for row in rows:
            haystack = f"{row['path']} {row['content']}".casefold()
            keyword_raw = sum(
                (3 if token in row["path"].casefold() else 1)
                * haystack.count(token)
                for token in query_tokens
            )
            highest_keyword_score = max(highest_keyword_score, keyword_raw)
            results.append({
                "path": row["path"],
                "chunk_index": row["chunk_index"],
                "content": row["content"],
                "semantic_score": float(row["semantic_score"]),
                "keyword_raw": keyword_raw,
            })

        for result in results:
            keyword_score = result.pop("keyword_raw") / highest_keyword_score
            result["keyword_score"] = keyword_score
            result["score"] = (
                SEMANTIC_WEIGHT * result["semantic_score"]
                + KEYWORD_WEIGHT * keyword_score
            )

        results.sort(
            key=lambda item: (
                -item["score"],
                item["path"],
                item["chunk_index"],
            )
        )
        return results[:limit]

    def status(self) -> Dict[str, Any]:
        with self._lock, self._connect() as connection:
            documents = connection.execute(
                "SELECT COUNT(*) AS count FROM documents "
                "WHERE owner_id = %s",
                (self.owner_id,),
            ).fetchone()["count"]
            chunks = connection.execute(
                "SELECT COUNT(*) AS count "
                "FROM chunks AS c "
                "JOIN documents AS d ON d.id = c.document_id "
                "WHERE d.owner_id = %s",
                (self.owner_id,),
            ).fetchone()["count"]
        return {
            "owner_id": self.owner_id,
            "folder": str(self.folder),
            "documents": documents,
            "chunks": chunks,
            "database": "PostgreSQL + pgvector",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
        }

    def watch(self) -> None: #2sn'de 1 kez selfsync() cagırarak klasörü kontrol eder
        while not self._stop.is_set():
            try:
                result = self.sync()
                if any(result.values()):
                    print(
                        f"[Library Connector kullanıcı={self.owner_id}] "
                        f"{result['added']} eklendi, "
                        f"{result['changed']} güncellendi, "
                        f"{result['deleted']} silindi"
                    )
            except Exception as exc:
                self.last_error = str(exc)
                print(
                    f"[Library Connector HATASI kullanıcı={self.owner_id}] {exc}"
                )
            self._stop.wait(LIBRARY_SCAN_SECONDS) #her library index kendi watch fonk. çalıştırır. 

    def start(self) -> None:
        threading.Thread(
            target=self.watch,
            name=f"library-connector-{self.owner_id}",
            daemon=True,
        ).start()


def load_library_indexes() -> Dict[str, LibraryIndex]:
#db'deki aktif kullanıcıları bulur, her kullancının kendi klasörü için ayrı bir LibraryIndex oluşturulmasına hazırlık yapar. 
    with psycopg.connect( #postegresql'e bağlanır aktif kullanıcıları çelker. 
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        users = connection.execute(
            """
            SELECT id, folder_path
            FROM users
            WHERE active = TRUE
              AND role = 'user'
            ORDER BY id
            """
        ).fetchall()  # Yönetici rolü için ayrı bir indeks oluşturulmaz.

    return {
        str(user["id"]): LibraryIndex(
            owner_id=user["id"],
            folder=Path(user["folder_path"]).expanduser(),
            database_url=DATABASE_URL,
        )
        for user in users
    }


library_indexes = load_library_indexes()

def get_library_members() -> Dict[str, str]: #kütüphanesi bulunan aktif kullanıcıları döndürür

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        rows = connection.execute(
            """
            SELECT id, display_name
            FROM users
            WHERE active = TRUE
              AND role = 'user'
            ORDER BY id
            """
        ).fetchall()

    return {
        str(row["id"]): row["display_name"]
        for row in rows
    }

def resolve_library_access(
    user,
    requested_scope: str,
) -> List[Tuple[str, str, LibraryIndex]]:

    # Giriş yapan kullanıcının erişebileceği kütüphaneleri belirler.

    members = get_library_members()

    # Normal kullanıcılar tarayıcıdan hangi kapsamı gönderirse göndersin sadece kendi kütüphanesini kullanabilir.
    if not user.is_admin: 
        user_library = library_indexes.get(user.id)

        if user_library is None:
            raise ValueError(
                "Bu kullanıcı için bir kütüphane tanımlanmamış."
            )

        return [
            (
                user.id,
                user.display_name,
                user_library,
            )
        ]

    # Yönetici "Tümü" seçtiğinde bütün kullanıcı indeksleri döner.
    if requested_scope in {"", "self", "all"}:
        selected_ids = list(members.keys())
    else:
        # Yönetici belirli bir kullanıcı seçti.
        if requested_scope not in members:
            raise ValueError(
                "Geçersiz kütüphane kapsamı."
            )

        selected_ids = [requested_scope]

    selected_libraries = []

    for user_id in selected_ids:
        library_index = library_indexes.get(user_id)

        if library_index is not None:
            selected_libraries.append(
                (
                    user_id,
                    members[user_id],
                    library_index,
                )
            )

    return selected_libraries

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


def search_libraries( #yönetici tümü seçtiğinde hem Ada hem Bob indexinde arama yapılır
    query: str,
    selected_libraries: List[
        Tuple[str, str, LibraryIndex]
    ],
    limit: int = 5,
) -> List[Dict[str, Any]]:
    """Seçilen kütüphanelerde arama yapıp sonuçları birleştirir."""

    combined_results: List[Dict[str, Any]] = []

    for owner_id, owner_name, library_index in selected_libraries:
        hits = library_index.search(
            query,
            limit=limit,
        )

        for hit in hits:
            combined_results.append({
                **hit,
                "owner_id": owner_id,
                "owner_name": owner_name,
            })

    combined_results.sort(
        key=lambda item: (
            -item["score"],
            item["owner_name"],
            item["path"],
            item["chunk_index"],
        )
    )

    return combined_results[:limit]

def run_agent(
    model: str,
    browser_messages: List[Dict[str, Any]],
    mode: str = "auto",
    user_libraries: Optional[
        List[Tuple[str, str, LibraryIndex]]
    ] = None,
    can_use_meter: bool = False,
) -> Tuple[str, List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    user_query = next((
        str(message.get("content", ""))
        for message in reversed(browser_messages)
        if message.get("role") == "user"
    ), "")
    normalized_query = user_query.casefold()
    thingsboard_terms = (
        "thingsboard", "cihaz", "telemetri", "sensör", "sensor",
        "hava kalitesi", "ölçüm", "sıcaklık", "sicaklik", "nem",
        "co₂", "co2", "pm2.5", "pm10", "voc", "aqi", "batarya",
        "list_air_quality_devices", "get_latest_air_quality",
        "get_device_context",
    )
    meter_terms = (
        "sayaç", "meter", "kwh", "kilovat", "kilowatt",
        "enerji tüketimi", "toplam enerji", "anlık güç",
        "aktif güç", "gerilim", "voltaj", "akım", "amper",
        "frekans", "güç faktörü", "power factor",
        "bağlantı durumu", "çevrim içi", "çevrimdışı",
        "list_meter_devices", "get_latest_meter_reading",
        "get_meter_context", "get_meter_history",
        "get_meter_energy_summary",
        "get_meter_connection_status",
    )
    has_meter_intent = any(
        term in normalized_query
        for term in meter_terms
    )

    is_meter_request = (
        can_use_meter
        and has_meter_intent
    )
    
    has_thingsboard_intent = any(
        term in normalized_query for term in thingsboard_terms
    )
    is_thingsboard_request = (
        "thingsboard" in normalized_query
        or "list_air_quality_devices" in normalized_query
        or "get_latest_air_quality" in normalized_query
        or "get_device_context" in normalized_query
    ) and any(word in normalized_query for word in ("cihaz", "hava", "ölçüm", "aqi"))

    if mode == "meter":
        if not can_use_meter:
            return (
                "Bu kullanıcının sayaç ajanına erişim izni yok.",
                events,
            )
        # Kullanıcı Sayaç modunu kendisi seçtiyse kısa sorular da geçerlidir.
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
        
        if is_meter_request:
            def call_meter_tool(
                name: str,
                arguments: Dict[str, Any],
            ) -> Dict[str, Any]:
                events.append({
                    "type": "tool_call",
                    "tool": name,
                    "arguments": arguments,
                })
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

            list_result = call_meter_tool(
                "list_meter_devices",
                {"page_size": 20, "search": ""},
            )
            error = tool_error_message(list_result)
            if error:
                return f"Sayaçlar ThingsBoard'dan alınamadı: {error}", events

            meters = result_data(list_result).get("meters", [])
            if not isinstance(meters, list) or not meters:
                return "ThingsBoard hesabında meter profiline ait sayaç bulunamadı.", events

            wants_list = any(word in normalized_query for word in (
                "listele", "listesi", "hangi sayaç", "kaç sayaç",
                "sayaclari", "sayaçları",
            ))
            wants_reading = any(word in normalized_query for word in (
                "güncel", "anlık", "değer", "ölçüm", "güç", "enerji",
                "gerilim", "voltaj", "akım", "frekans", "bağlantı",
            ))

            if wants_list and not wants_reading:
                lines = [f"ThingsBoard hesabında {len(meters)} sayaç bulundu:"]
                for index, meter in enumerate(meters, 1):
                    lines.append(
                        f"{index}. {meter.get('name', 'Adsız sayaç')} "
                        f"(ID: {meter.get('device_id', '-')})"
                    )
                return "\n".join(lines), events

            meter = meters[0]
            meter_name = str(meter.get("name", "Sayaç"))
            device_id = str(meter.get("device_id", ""))
            if not device_id:
                return "Sayaç kimliği ThingsBoard yanıtında bulunamadı.", events

            wants_context = any(word in normalized_query for word in (
                "konum", "bina", "kat", "model", "seri numarası",
                "seri no", "firmware", "nerede",
            ))
            wants_connection = any(word in normalized_query for word in (
                "çevrim içi", "çevrimiçi", "çevrim dışı", "offline",
                "online", "son telemetri", "bağlantı durumu",
                "bağlantı bilgi", "son veri", "ne zaman veri",
                "veri ne zaman", "veri gelmiş",
            ))
            wants_history = any(word in normalized_query for word in (
                "geçmiş", "değişmiş", "değişim", "son bir saat",
                "son 24 saat", "grafik", "trend",
            ))
            wants_summary = any(word in normalized_query for word in (
                "bugün tüketti", "bugünkü tüketim", "tüketim kaç",
                "tükettiği enerji", "enerji özeti",
            ))

            context_labels = (
                ("building", "Bina", ("bina",)),
                ("floor", "Kat", ("kat",)),
                ("location", "Konum", ("konum", "nerede")),
                ("model", "Model", ("model",)),
                ("firmwareVersion", "Firmware", ("firmware",)),
                ("meterType", "Sayaç tipi", ("sayaç tipi", "faz")),
                ("serialNumber", "Seri numarası", ("seri numarası", "seri no")),
            )
            requested_context_keys = [
                key
                for key, _label, phrases in context_labels
                if any(phrase in normalized_query for phrase in phrases)
            ]

            # Kullanıcı yalnızca "cihaz bilgileri" dediyse tüm bağlamı göster.
            if wants_context and not requested_context_keys:
                requested_context_keys = [
                    key for key, _label, _phrases in context_labels
                ]

            if wants_context:
                context_result = call_meter_tool(
                    "get_meter_context",
                    {"device_id": device_id},
                )
                error = tool_error_message(context_result)
                if error:
                    return f"Sayaç bağlam bilgileri alınamadı: {error}", events

                attributes = result_data(context_result).get("attributes", {})
                if not isinstance(attributes, dict):
                    attributes = {}
                lines = [f"{meter_name} cihaz bilgileri:"]
                for key, label, _phrases in context_labels:
                    if key not in requested_context_keys:
                        continue
                    value = attributes.get(key)
                    if value is not None:
                        lines.append(f"• {label}: {value}")
                if len(lines) == 1:
                    lines.append("• Cihaz için ek bağlam bilgisi tanımlanmamış.")
                return "\n".join(lines), events

            if wants_connection:
                connection_result = call_meter_tool(
                    "get_meter_connection_status",
                    {"device_id": device_id},
                )
                error = tool_error_message(connection_result)
                if error:
                    return f"Bağlantı durumu alınamadı: {error}", events

                status = result_data(connection_result)
                reported_status = status.get(
                    "reported_status",
                    "bilinmiyor",
                )
                last_telemetry_at = status.get(
                    "last_telemetry_at",
                    "bilinmiyor",
                )
                freshness = status.get(
                    "freshness",
                    "Veri güncelliği hesaplanamadı.",
                )

                return (
                    f"{meter_name} bağlantı bilgisi:\n"
                    f"• Cihazın son bildirdiği durum: {reported_status}\n"
                    f"• Son telemetri zamanı: {last_telemetry_at}\n"
                    f"• Veri güncelliği: {freshness}",
                    events,
                )

            now_ms = int(time.time() * 1000)

            if wants_summary:
                today_start = datetime.now().replace(
                    hour=0,
                    minute=0,
                    second=0,
                    microsecond=0,
                )
                summary_result = call_meter_tool(
                    "get_meter_energy_summary",
                    {
                        "device_id": device_id,
                        "start_ts": int(today_start.timestamp() * 1000),
                        "end_ts": now_ms,
                    },
                )
                error = tool_error_message(summary_result)
                if error:
                    return f"Enerji özeti alınamadı: {error}", events

                summary = result_data(summary_result)
                if summary.get("error"):
                    return str(summary["error"]), events
                return (
                    f"{meter_name} bugünkü enerji özeti:\n"
                    f"• Başlangıç: {summary.get('start_energy_kwh', '-')} kWh\n"
                    f"• Güncel: {summary.get('end_energy_kwh', '-')} kWh\n"
                    f"• Tüketim: {summary.get('consumption_kwh', '-')} kWh",
                    events,
                )

            if wants_history:
                hours = 24 if "24" in normalized_query else 1
                metric = "voltage_v" if any(word in normalized_query for word in (
                    "gerilim", "voltaj",
                )) else "active_power_kw"
                history_result = call_meter_tool(
                    "get_meter_history",
                    {
                        "device_id": device_id,
                        "metric": metric,
                        "start_ts": now_ms - (hours * 60 * 60 * 1000),
                        "end_ts": now_ms,
                    },
                )
                error = tool_error_message(history_result)
                if error:
                    return f"Sayaç geçmişi alınamadı: {error}", events

                history = result_data(history_result)
                points = history.get("points", [])
                if not points:
                    return (
                        f"Son {hours} saat için {metric} geçmiş verisi bulunamadı.",
                        events,
                    )
                return (
                    f"{meter_name} için son {hours} saatlik {metric} geçmişi:\n"
                    f"• Veri noktası sayısı: {len(points)}\n"
                    f"• İlk değer: {points[0].get('value', '-')}\n"
                    f"• Son değer: {points[-1].get('value', '-')}",
                    events,
                )

            reading_result = call_meter_tool(
                "get_latest_meter_reading",
                {"device_id": device_id},
            )
            error = tool_error_message(reading_result)
            if error:
                return f"Sayaç verileri alınamadı: {error}", events

            measurements = result_data(reading_result).get("measurements", {})
            if not isinstance(measurements, dict):
                measurements = {}

            def measurement_value(key: str) -> str:
                item = measurements.get(key)
                if isinstance(item, dict):
                    return str(item.get("value", "-"))
                return "-"

            measurement_labels = (
                ("energy_total_kwh", "Toplam enerji", "kWh", ("enerji", "kwh")),
                ("active_power_kw", "Anlık aktif güç", "kW", ("güç", "kw")),
                ("voltage_v", "Gerilim", "V", ("gerilim", "voltaj")),
                ("current_a", "Akım", "A", ("akım", "amper")),
                ("frequency_hz", "Frekans", "Hz", ("frekans", "hz")),
                ("power_factor", "Güç faktörü", "", ("güç faktörü", "cos")),
                ("connection_status", "Bağlantı durumu", "", ("bağlantı", "online", "offline")),
            )
            requested_measurement_keys = [
                key
                for key, _label, _unit, phrases in measurement_labels
                if any(phrase in normalized_query for phrase in phrases)
            ]
            if not requested_measurement_keys:
                requested_measurement_keys = [
                    key for key, _label, _unit, _phrases in measurement_labels
                ]

            lines = [f"{meter_name} güncel değerleri:"]
            for key, label, unit, _phrases in measurement_labels:
                if key not in requested_measurement_keys:
                    continue
                suffix = f" {unit}" if unit else ""
                lines.append(
                    f"• {label}: {measurement_value(key)}{suffix}"
                )
            return "\n".join(lines), events

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
    if current_user.is_authenticated:
        return redirect(url_for("chat_page"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        user = authenticate_user(username, password)

        if user is None:
            error = "Kullanıcı adı veya parola hatalı."
        else:
            login_user(user)
            return redirect(url_for("chat_page"))

    return render_template(
        "login.html",
        error=error,
    )


@app.post("/logout")
@login_required
def logout():
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

        model = body.get("model", "gemma3:4b")
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
        if mode == "meter" and not current_user.can_use_meter:
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
        )

        return jsonify({
            "content": content,
            "events": events,
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
