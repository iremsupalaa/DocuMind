"""Kullanıcı veritabanı sorguları."""

import os
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql:///ollama_library",
)


def get_user_row_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Kullanıcıyı veritabanı kimliğiyle getirir."""

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        return connection.execute(
            """
            SELECT
                id,
                username,
                display_name,
                password_hash,
                folder_path,
                active,
                role,
                meter_access
            FROM users
            WHERE id = %s
            """,
            (user_id,),
        ).fetchone()


def get_user_row_by_username(
    username: str,
) -> Optional[Dict[str, Any]]:
    """Kullanıcıyı kullanıcı adıyla getirir."""

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        return connection.execute(
            """
            SELECT
                id,
                username,
                display_name,
                password_hash,
                folder_path,
                active,
                role,
                meter_access
            FROM users
            WHERE username = %s
            """,
            (username,),
        ).fetchone()


def update_user_password_hash(
    user_id: int,
    password_hash: str,
) -> None:
    """Kullanıcının parola özetini günceller."""

    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            """,
            (password_hash, user_id),
        )


def get_active_user_rows() -> List[Dict[str, Any]]:
    """Aktif kullanıcıları klasör bilgileriyle listeler."""

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        return connection.execute(
            """
            SELECT
                id,
                username,
                display_name,
                folder_path,
                active,
                role,
                meter_access
            FROM users
            WHERE active = TRUE
            ORDER BY id
            """
        ).fetchall()