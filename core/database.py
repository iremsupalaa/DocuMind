"""PostgreSQL bağlantı havuzu yönetimi."""

import atexit
import os
from contextlib import contextmanager
from typing import Iterator

from pgvector.psycopg import register_vector
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from core.config import DATABASE_URL


DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
DB_POOL_TIMEOUT = float(os.getenv("DB_POOL_TIMEOUT", "5"))

if DB_POOL_MAX_SIZE < DB_POOL_MIN_SIZE:
    raise RuntimeError(
        "DB_POOL_MAX_SIZE, DB_POOL_MIN_SIZE değerinden küçük olamaz."
    )


def configure_connection(connection: Connection) -> None:
    """Havuza eklenen her bağlantıda pgvector tipini hazırlar."""

    connection.autocommit = True

    try:
        register_vector(connection)
    finally:
        connection.autocommit = False


connection_pool = ConnectionPool(
    conninfo=DATABASE_URL,
    min_size=DB_POOL_MIN_SIZE,
    max_size=DB_POOL_MAX_SIZE,
    timeout=DB_POOL_TIMEOUT,
    max_idle=300,
    kwargs={
        "row_factory": dict_row,
    },
    configure=configure_connection,
    open=True,
)


@contextmanager
def get_connection() -> Iterator[Connection]:
    """Havuzdan bağlantı alır, iş bitince tekrar havuza bırakır."""

    with connection_pool.connection() as connection:
        yield connection


@contextmanager
def get_autocommit_connection() -> Iterator[Connection]:
    """Şema/DDL işlemleri için autocommit bağlantısı verir."""

    with connection_pool.connection() as connection:
        connection.autocommit = True

        try:
            yield connection
        finally:
            connection.autocommit = False


def close_connection_pool() -> None:
    """Uygulama kapanırken havuz bağlantılarını kapatır."""

    connection_pool.close()


atexit.register(close_connection_pool)