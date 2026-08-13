#!/usr/bin/env python3
"""
->users tablosunu oluşturur
-> kullanıcı klasörünü oluşturur
-> parolayı terminalde göstermeden alır
-> parolayı argon2 ile hashler
-> db'ye düz şifre kaydetmez
-> kullancının LC klasörü dışındaki bir klasöre bağlanmasını engeller


"""
import argparse
import os
from getpass import getpass
from pathlib import Path

import psycopg
from argon2 import PasswordHasher


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql:///ollama_library",
)

LIBRARY_ROOT = (
    Path.home()
    / "Desktop"
    / "Library-Connector"
)

password_hasher = PasswordHasher()


def create_users_table(connection):
    """Kullanıcı tablosunu mevcut değilse oluşturur."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id BIGSERIAL PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            display_name TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            folder_path TEXT UNIQUE NOT NULL,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            role TEXT NOT NULL DEFAULT 'user',
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def validate_folder_name(folder_name):
    """Klasör adının Library-Connector dışına çıkmasını engeller."""

    folder_name = folder_name.strip()

    if not folder_name:
        raise ValueError("Klasör adı boş olamaz.")

    library_root = LIBRARY_ROOT.resolve()
    folder_path = (library_root / folder_name).resolve()

    if folder_path.parent != library_root:
        raise ValueError(
            "Klasör doğrudan Library-Connector altında olmalıdır."
        )

    return folder_path


def read_password():
    """Parolayı terminalde görünmeden iki defa ister."""

    password = getpass("Parola: ")
    password_confirmation = getpass("Parolayı tekrar girin: ")

    if password != password_confirmation:
        raise ValueError("Girilen parolalar eşleşmiyor.")

    if len(password) < 8:
        raise ValueError("Parola en az 8 karakter olmalıdır.")

    return password


def save_user(username, display_name, folder_path, password):
    """Kullanıcıyı PostgreSQL veritabanına kaydeder."""

    password_hash = password_hasher.hash(password)

    with psycopg.connect(DATABASE_URL) as connection:
        create_users_table(connection)

        row = connection.execute(
            """
            INSERT INTO users (
                username,
                display_name,
                password_hash,
                folder_path,
                active
            )
            VALUES (%s, %s, %s, %s, TRUE)
            ON CONFLICT (username)
            DO UPDATE SET
                display_name = EXCLUDED.display_name,
                password_hash = EXCLUDED.password_hash,
                folder_path = EXCLUDED.folder_path,
                active = TRUE
            RETURNING id
            """,
            (
                username,
                display_name,
                password_hash,
                str(folder_path),
            ),
        ).fetchone()

    return row[0]


def main():
    parser = argparse.ArgumentParser(
        description="Ollama uygulamasına kullanıcı ekler."
    )

    parser.add_argument(
        "username",
        help="Girişte kullanılacak kullanıcı adı",
    )

    parser.add_argument(
        "display_name",
        help="Ekranda gösterilecek kullanıcı adı",
    )

    parser.add_argument(
        "folder_name",
        help="Library-Connector altındaki klasör adı",
    )

    arguments = parser.parse_args()

    username = arguments.username.strip().lower()
    display_name = arguments.display_name.strip()

    if not username:
        raise ValueError("Kullanıcı adı boş olamaz.")

    if not display_name:
        raise ValueError("Görünen ad boş olamaz.")

    folder_path = validate_folder_name(arguments.folder_name)

    LIBRARY_ROOT.mkdir(parents=True, exist_ok=True)
    folder_path.mkdir(parents=True, exist_ok=True)

    password = read_password()

    user_id = save_user(
        username=username,
        display_name=display_name,
        folder_path=folder_path,
        password=password,
    )

    print()
    print("Kullanıcı kaydedildi.")
    print(f"ID: {user_id}")
    print(f"Kullanıcı adı: {username}")
    print(f"Görünen ad: {display_name}")
    print(f"Klasör: {folder_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"[HATA] {error}")
        raise SystemExit(1)