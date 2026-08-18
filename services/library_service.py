import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from core.config import (
    DATABASE_URL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LIBRARY_SCAN_SECONDS,
)
from services.ollama_service import ollama_embeddings

SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
library_indexes: Dict[str, "LibraryIndex"] = {}

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


def set_library_indexes(indexes: Dict[str, LibraryIndex]) -> None:
    """Uygulama başlangıcında oluşturulan indeksleri servis içinde saklar."""
    global library_indexes
    library_indexes = indexes

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
