"""Belge ve parça veritabanı sorguları."""

from typing import Any, Dict, List

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row


def initialize_library_schema(
    database_url: str,
    embedding_dim: int,
) -> None:
    """pgvector eklentisini, belge ve parça tablolarını hazırlar."""

    with psycopg.connect(
        database_url,
        autocommit=True,
        row_factory=dict_row,
    ) as connection:
        connection.execute(
            "CREATE EXTENSION IF NOT EXISTS vector"
        )
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
                embedding vector({embedding_dim}) NOT NULL,
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


def get_documents_by_owner(
    connection: psycopg.Connection,
    owner_id: int,
) -> Dict[str, Dict[str, Any]]:
    """Kullanıcının indekslenmiş belgelerini yol anahtarıyla döndürür."""

    rows = connection.execute(
        """
        SELECT
            id,
            path,
            mtime_ns,
            size,
            sha256,
            embedding_model,
            embedding_dim
        FROM documents
        WHERE owner_id = %s
        """,
        (owner_id,),
    ).fetchall()

    return {
        row["path"]: row
        for row in rows
    }


def update_document_file_metadata(
    connection: psycopg.Connection,
    document_id: int,
    owner_id: int,
    mtime_ns: int,
    size: int,
) -> None:
    """İçeriği aynı kalan belgenin dosya zamanını ve boyutunu günceller."""

    connection.execute(
        """
        UPDATE documents
        SET mtime_ns = %s, size = %s
        WHERE id = %s AND owner_id = %s
        """,
        (mtime_ns, size, document_id, owner_id),
    )


def delete_document(
    connection: psycopg.Connection,
    document_id: int,
    owner_id: int,
) -> None:
    """Belgeyi ve bağlı parçalarını siler."""

    connection.execute(
        """
        DELETE FROM documents
        WHERE id = %s AND owner_id = %s
        """,
        (document_id, owner_id),
    )


def create_document(
    connection: psycopg.Connection,
    *,
    owner_id: int,
    path: str,
    mtime_ns: int,
    size: int,
    sha256: str,
    embedding_model: str,
    embedding_dim: int,
    updated_at: float,
) -> int:
    """Yeni belge kaydı oluşturur ve kimliğini döndürür."""

    row = connection.execute(
        """
        INSERT INTO documents (
            owner_id,
            path,
            mtime_ns,
            size,
            sha256,
            embedding_model,
            embedding_dim,
            updated_at
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            owner_id,
            path,
            mtime_ns,
            size,
            sha256,
            embedding_model,
            embedding_dim,
            updated_at,
        ),
    ).fetchone()

    return int(row["id"])


def create_chunks(
    connection: psycopg.Connection,
    document_id: int,
    chunks: List[str],
    embeddings: List[List[float]],
) -> None:
    """Belgenin metin parçalarını ve vektörlerini kaydeder."""

    if not chunks:
        return

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO chunks (
                document_id,
                chunk_index,
                content,
                embedding
            )
            VALUES (%s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    chunk_index,
                    chunk,
                    Vector(embedding),
                )
                for chunk_index, (chunk, embedding)
                in enumerate(zip(chunks, embeddings))
            ],
        )


def search_document_chunks(
    connection: psycopg.Connection,
    *,
    owner_id: int,
    query_vector: Vector,
    limit: int,
) -> List[Dict[str, Any]]:
    """Kullanıcının parçalarında pgvector kosinüs araması yapar."""

    return connection.execute(
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
            owner_id,
            query_vector,
            limit,
        ),
    ).fetchall()


def get_library_counts(
    connection: psycopg.Connection,
    owner_id: int,
) -> Dict[str, int]:
    """Kullanıcının belge ve parça sayılarını döndürür."""

    document_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM documents
        WHERE owner_id = %s
        """,
        (owner_id,),
    ).fetchone()["count"]

    chunk_count = connection.execute(
        """
        SELECT COUNT(*) AS count
        FROM chunks AS c
        JOIN documents AS d ON d.id = c.document_id
        WHERE d.owner_id = %s
        """,
        (owner_id,),
    ).fetchone()["count"]

    return {
        "documents": int(document_count),
        "chunks": int(chunk_count),
    }


def get_active_library_users(
    database_url: str,
) -> List[Dict[str, Any]]:
    """Kütüphanesi olan aktif normal kullanıcıları listeler."""

    with psycopg.connect(
        database_url,
        row_factory=dict_row,
    ) as connection:
        return connection.execute(
            """
            SELECT id, display_name, folder_path
            FROM users
            WHERE active = TRUE
              AND role = 'user'
            ORDER BY id
            """
        ).fetchall()