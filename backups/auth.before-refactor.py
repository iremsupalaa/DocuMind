#!/usr/bin/env python3

"""Kullanıcı bulma ve parola doğrulama işlemleri."""

import os
from typing import Optional

import psycopg
from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from flask_login import UserMixin
from psycopg.rows import dict_row


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql:///ollama_library",
)

password_hasher = PasswordHasher()


class User(UserMixin):
    """Flask-Login tarafından kullanılacak kullanıcı nesnesi."""

    def __init__(
        self,
        user_id: int,
        username: str,
        display_name: str,
        password_hash: str,
        folder_path: str,
        active: bool,
        role: str,
        meter_access: bool,
    ):
        self.id = str(user_id)
        self.username = username
        self.display_name = display_name
        self.password_hash = password_hash
        self.folder_path = folder_path
        self.active = active
        self.role = role
        self.meter_access = meter_access

    @property
    def is_active(self) -> bool:
        return bool(self.active)

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_use_meter(self) -> bool: #kullanıcının meter agent'ına erişim izni var mı?
        return bool(self.meter_access)


def row_to_user(row) -> Optional[User]:
    """PostgreSQL satırını User nesnesine dönüştürür."""

    if row is None:
        return None

    return User(
        user_id=row["id"],
        username=row["username"],
        display_name=row["display_name"],
        password_hash=row["password_hash"],
        folder_path=row["folder_path"],
        active=row["active"],
        role=row["role"],
        meter_access=row["meter_access"],
    )


def get_user_by_id(user_id: str) -> Optional[User]:
    """Kullanıcıyı veritabanı kimliğiyle bulur."""

    try:
        numeric_user_id = int(user_id)
    except (TypeError, ValueError):
        return None

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        row = connection.execute(
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
            (numeric_user_id,),
        ).fetchone()

    return row_to_user(row)


def get_user_by_username(username: str) -> Optional[User]:
    """Kullanıcıyı kullanıcı adıyla bulur."""

    normalized_username = username.strip().lower()

    if not normalized_username:
        return None

    with psycopg.connect(
        DATABASE_URL,
        row_factory=dict_row,
    ) as connection:
        row = connection.execute(
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
            (normalized_username,),
        ).fetchone()

    return row_to_user(row)


def update_password_hash(user_id: str, password: str) -> None:
    """Eski Argon2 ayarlarıyla üretilen özeti gerektiğinde yeniler."""

    new_password_hash = password_hasher.hash(password)

    with psycopg.connect(DATABASE_URL) as connection:
        connection.execute(
            """
            UPDATE users
            SET password_hash = %s
            WHERE id = %s
            """,
            (
                new_password_hash,
                int(user_id),
            ),
        )


def authenticate_user(
    username: str,
    password: str,
) -> Optional[User]:
    """Kullanıcı adı ve parolayı doğrular."""

    user = get_user_by_username(username)

    if user is None or not user.is_active:
        return None

    try:
        password_hasher.verify(
            user.password_hash,
            password,
        )
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return None

    if password_hasher.check_needs_rehash(
        user.password_hash
    ):
        update_password_hash(user.id, password)
        user = get_user_by_id(user.id)

    return user