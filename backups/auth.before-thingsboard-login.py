#!/usr/bin/env python3
"""Kullanıcı bulma ve parola doğrulama işlemleri."""

from typing import Any, Dict, Optional

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)
from flask_login import UserMixin

from repositories.user_repository import (
    get_user_row_by_id,
    get_user_row_by_username,
    update_user_password_hash,
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
        """Kullanıcının hesabının aktif olup olmadığını döndürür."""
        return bool(self.active)

    @property
    def is_admin(self) -> bool:
        """Kullanıcının yönetici rolünde olup olmadığını döndürür."""
        return self.role == "admin"

    @property
    def can_use_meter(self) -> bool:
        """Kullanıcının sayaç ajanına erişim iznini döndürür."""
        return bool(self.meter_access)


def row_to_user(row: Optional[Dict[str, Any]]) -> Optional[User]:
    """Veritabanı satırını Flask-Login User nesnesine dönüştürür."""

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
        meter_access=row.get("meter_access", False),
    )


def get_user_by_id(user_id: str) -> Optional[User]:
    """Kullanıcıyı veritabanı kimliğiyle bulur."""

    try:
        numeric_user_id = int(user_id)
    except (TypeError, ValueError):
        return None

    return row_to_user(get_user_row_by_id(numeric_user_id))


def get_user_by_username(username: str) -> Optional[User]:
    """Kullanıcıyı kullanıcı adıyla bulur."""

    normalized_username = username.strip().lower()

    if not normalized_username:
        return None

    return row_to_user(
        get_user_row_by_username(normalized_username)
    )


def update_password_hash(user_id: str, password: str) -> None:
    """Parola özetini güncel Argon2 ayarlarıyla yeniler."""

    new_password_hash = password_hasher.hash(password)

    update_user_password_hash(
        user_id=int(user_id),
        password_hash=new_password_hash,
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