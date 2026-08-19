"""Kullanıcı veritabanı sorguları."""

import os
from typing import Any, Dict, List, Optional

import psycopg
from psycopg.rows import dict_row


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql:///ollama_library",
)


def ensure_thingsboard_identity_schema() -> None:
    """ThingsBoard e-posta eşleştirmesi için gereken alanı hazırlar."""

    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            ALTER TABLE users
            ADD COLUMN IF NOT EXISTS thingsboard_email TEXT
            """
        )
        connection.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS
            users_thingsboard_email_unique
            ON users (LOWER(thingsboard_email))
            WHERE thingsboard_email IS NOT NULL
            """
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
                meter_access,
                thingsboard_email
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
                meter_access,
                thingsboard_email
            FROM users
            WHERE username = %s
            """,
            (username,),
        ).fetchone()


def get_user_row_by_thingsboard_email(
    email: str,
) -> Optional[Dict[str, Any]]:
    """ThingsBoard e-postasıyla eşleşen yerel kullanıcıyı getirir."""

    normalized_email = email.strip().lower()

    if not normalized_email:
        return None

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
                meter_access,
                thingsboard_email
            FROM users
            WHERE LOWER(thingsboard_email) = %s
            """,
            (normalized_email,),
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
                meter_access,
                thingsboard_email
            FROM users
            WHERE active = TRUE
            ORDER BY id
            """
        ).fetchall()

