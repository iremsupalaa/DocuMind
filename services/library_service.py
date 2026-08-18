"""Kullanıcı kütüphanelerini tarayan ve pgvector araması yapan servis."""

import hashlib
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
from repositories.library_repository import (
    create_chunks,
    create_document,
    delete_document,
    get_active_library_users,
    get_documents_by_owner,
    get_library_counts,
    initialize_library_schema,
    search_document_chunks,
    update_document_file_metadata,
)
from services.ollama_service import ollama_embeddings


SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3

library_indexes: Dict[str, "LibraryIndex"] = {}


class LibraryIndex:
    """Yerel metin dosyalarını PostgreSQL + pgvector ile eşitler."""

    SUPPORTED_SUFFIXES = {".txt", ".md"}

    STOP_WORDS = {
        "acaba", "ama", "bana", "ben", "bir", "bu", "da", "de", "diye",
        "en", "gibi", "hakkında", "hangi", "ile", "için", "kim", "mi",
        "mı", "mu", "mü", "nasıl", "ne", "nedir", "neden", "olan",
        "olarak", "onu", "şu", "ve", "veya",
    }

    def __init__(
        self,
        owner_id: int,
        folder: Path,
        database_url: str,
    ):
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
        """pgvector kayıtlı PostgreSQL bağlantısı oluşturur."""

        connection = psycopg.connect(
            self.database_url,
            row_factory=dict_row,
        )
        register_vector(connection)
        return connection

    def _initialize_database(self) -> None:
        """Tablo ve indeks hazırlama işini repository katmanına devreder."""

        initialize_library_schema(
            database_url=self.database_url,
            embedding_dim=EMBEDDING_DIM,
        )

    @staticmethod
    def _chunks(
        text: str,
        target_size: int = 1200,
        overlap: int = 180,
    ) -> List[str]:
        """Metni örtüşen, arama için uygun parçalara böler."""

        normalized = re.sub(r"\r\n?", "\n", text).strip()

        if not normalized:
            return []

        chunks: List[str] = []
        start = 0

        while start < len(normalized):
            end = min(start + target_size, len(normalized))

            if end < len(normalized):
                boundary = max(
                    normalized.rfind(
                        "\n",
                        start + target_size // 2,
                        end,
                    ),
                    normalized.rfind(
                        ". ",
                        start + target_size // 2,
                        end,
                    ),
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
        """Eklenen/değişen belgeleri indeksler, silinenleri kaldırır."""

        with self._lock:
            discovered: Dict[str, Path] = {}

            for file_path in self.folder.rglob("*"):
                if (
                    file_path.is_file()
                    and file_path.suffix.casefold()
                    in self.SUPPORTED_SUFFIXES
                ):
                    relative_path = str(
                        file_path.relative_to(self.folder)
                    )
                    discovered[relative_path] = file_path

            added = 0
            changed = 0
            deleted = 0

            with self._connect() as connection:
                stored = get_documents_by_owner(
                    connection=connection,
                    owner_id=self.owner_id,
                )

                for relative_path, file_path in discovered.items():
                    stat = file_path.stat()
                    previous = stored.get(relative_path)

                    content = file_path.read_text(
                        encoding="utf-8",
                        errors="replace",
                    )

                    digest = hashlib.sha256(
                        content.encode("utf-8")
                    ).hexdigest()

                    is_same_content = (
                        previous
                        and previous["sha256"] == digest
                        and previous["embedding_model"]
                        == EMBEDDING_MODEL
                        and previous["embedding_dim"]
                        == EMBEDDING_DIM
                    )

                    if is_same_content:
                        file_metadata_changed = (
                            previous["mtime_ns"] != stat.st_mtime_ns
                            or previous["size"] != stat.st_size
                        )

                        if file_metadata_changed:
                            update_document_file_metadata(
                                connection=connection,
                                document_id=previous["id"],
                                owner_id=self.owner_id,
                                mtime_ns=stat.st_mtime_ns,
                                size=stat.st_size,
                            )

                        continue

                    chunks = self._chunks(content)
                    embeddings = ollama_embeddings(chunks)

                    if len(chunks) != len(embeddings):
                        raise ValueError(
                            "Parça ve embedding sayıları eşleşmedi."
                        )

                    if any(
                        len(embedding) != EMBEDDING_DIM
                        for embedding in embeddings
                    ):
                        raise ValueError(
                            f"Embedding boyutu "
                            f"{EMBEDDING_DIM} olmalıdır."
                        )

                    if previous:
                        delete_document(
                            connection=connection,
                            document_id=previous["id"],
                            owner_id=self.owner_id,
                        )

                    document_id = create_document(
                        connection=connection,
                        owner_id=self.owner_id,
                        path=relative_path,
                        mtime_ns=stat.st_mtime_ns,
                        size=stat.st_size,
                        sha256=digest,
                        embedding_model=EMBEDDING_MODEL,
                        embedding_dim=EMBEDDING_DIM,
                        updated_at=time.time(),
                    )

                    create_chunks(
                        connection=connection,
                        document_id=document_id,
                        chunks=chunks,
                        embeddings=embeddings,
                    )

                    if previous:
                        changed += 1
                    else:
                        added += 1

                for relative_path in set(stored) - set(discovered):
                    delete_document(
                        connection=connection,
                        document_id=stored[relative_path]["id"],
                        owner_id=self.owner_id,
                    )
                    deleted += 1

            self.last_sync_at = time.time()
            self.last_error = None

            return {
                "added": added,
                "changed": changed,
                "deleted": deleted,
            }

    @staticmethod
    def _tokens(text: str) -> List[str]:
        """Sorgudan anlamlı anahtar kelimeleri çıkarır."""

        return [
            token
            for token in re.findall(
                r"[\wçğıöşüÇĞİÖŞÜ]{3,}",
                text.casefold(),
            )
            if token not in LibraryIndex.STOP_WORDS
        ]

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """pgvector ve anahtar kelime puanlarını birleştirerek arar."""

        self.sync()

        query_embedding = ollama_embeddings([query])[0]

        if len(query_embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"Sorgu embedding boyutu "
                f"{EMBEDDING_DIM} olmalıdır."
            )

        query_vector = Vector(query_embedding)
        query_tokens = self._tokens(query)
        candidate_limit = max(limit * 10, 50)

        with self._lock, self._connect() as connection:
            rows = search_document_chunks(
                connection=connection,
                owner_id=self.owner_id,
                query_vector=query_vector,
                limit=candidate_limit,
            )

        if not rows:
            return []

        results: List[Dict[str, Any]] = []
        highest_keyword_score = 1

        for row in rows:
            haystack = (
                f"{row['path']} {row['content']}"
            ).casefold()

            keyword_raw = sum(
                (
                    3
                    if token in row["path"].casefold()
                    else 1
                )
                * haystack.count(token)
                for token in query_tokens
            )

            highest_keyword_score = max(
                highest_keyword_score,
                keyword_raw,
            )

            results.append(
                {
                    "path": row["path"],
                    "chunk_index": row["chunk_index"],
                    "content": row["content"],
                    "semantic_score": float(
                        row["semantic_score"]
                    ),
                    "keyword_raw": keyword_raw,
                }
            )

        for result in results:
            keyword_score = (
                result.pop("keyword_raw")
                / highest_keyword_score
            )

            result["keyword_score"] = keyword_score
            result["score"] = (
                SEMANTIC_WEIGHT
                * result["semantic_score"]
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
        """Kullanıcının indeks istatistiklerini döndürür."""

        with self._lock, self._connect() as connection:
            counts = get_library_counts(
                connection=connection,
                owner_id=self.owner_id,
            )

        return {
            "owner_id": self.owner_id,
            "folder": str(self.folder),
            "documents": counts["documents"],
            "chunks": counts["chunks"],
            "database": "PostgreSQL + pgvector",
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "last_sync_at": self.last_sync_at,
            "last_error": self.last_error,
        }

    def watch(self) -> None:
        """Klasörü belirtilen saniye aralığıyla kontrol eder."""

        while not self._stop.is_set():
            try:
                result = self.sync()

                if any(result.values()):
                    print(
                        f"[Library Connector "
                        f"kullanıcı={self.owner_id}] "
                        f"{result['added']} eklendi, "
                        f"{result['changed']} güncellendi, "
                        f"{result['deleted']} silindi"
                    )

            except Exception as error:
                self.last_error = str(error)

                print(
                    f"[Library Connector HATASI "
                    f"kullanıcı={self.owner_id}] {error}"
                )

            self._stop.wait(LIBRARY_SCAN_SECONDS)

    def start(self) -> None:
        """Kütüphane izleme iş parçacığını başlatır."""

        threading.Thread(
            target=self.watch,
            name=f"library-connector-{self.owner_id}",
            daemon=True,
        ).start()


def load_library_indexes() -> Dict[str, LibraryIndex]:
    """Aktif kullanıcılar için ayrı kütüphane indeksleri oluşturur."""

    users = get_active_library_users(DATABASE_URL)

    return {
        str(user["id"]): LibraryIndex(
            owner_id=user["id"],
            folder=Path(user["folder_path"]).expanduser(),
            database_url=DATABASE_URL,
        )
        for user in users
    }


def set_library_indexes(
    indexes: Dict[str, LibraryIndex],
) -> None:
    """Uygulama başlangıcında oluşturulan indeksleri servis içinde saklar."""

    global library_indexes
    library_indexes = indexes


def get_library_members() -> Dict[str, str]:
    """Kütüphanesi bulunan aktif kullanıcıları döndürür."""

    users = get_active_library_users(DATABASE_URL)

    return {
        str(user["id"]): user["display_name"]
        for user in users
    }


def resolve_library_access(
    user: Any,
    requested_scope: str,
) -> List[Tuple[str, str, LibraryIndex]]:
    """Giriş yapan kullanıcının erişebileceği kütüphaneleri belirler."""

    members = get_library_members()

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

    if requested_scope in {"", "self", "all"}:
        selected_ids = list(members.keys())
    else:
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


def search_libraries(
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
            query=query,
            limit=limit,
        )

        for hit in hits:
            combined_results.append(
                {
                    **hit,
                    "owner_id": owner_id,
                    "owner_name": owner_name,
                }
            )

    combined_results.sort(
        key=lambda item: (
            -item["score"],
            item["owner_name"],
            item["path"],
            item["chunk_index"],
        )
    )

    return combined_results[:limit]