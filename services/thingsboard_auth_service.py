"""ThingsBoard üzerinden kullanıcı kimliği doğrulama servisi."""

from __future__ import annotations

import json
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from core.config import (
    THINGSBOARD_AUTH_TIMEOUT_SECONDS,
    THINGSBOARD_URL,
)


class ThingsBoardAuthError(Exception):
    """ThingsBoard kimlik doğrulama hatası."""


@dataclass(frozen=True)
class ThingsBoardUser:
    """ThingsBoard tarafından döndürülen kullanıcı bilgileri."""

    user_id: str
    email: str
    first_name: str
    last_name: str
    authority: str
    tenant_id: Optional[str] = None
    customer_id: Optional[str] = None

    @property
    def display_name(self) -> str:
        """Kullanıcının ekranda gösterilecek adını döndürür."""

        full_name = " ".join(
            item.strip()
            for item in (
                self.first_name,
                self.last_name,
            )
            if item and item.strip()
        )

        return full_name or self.email


@dataclass(frozen=True)
class ThingsBoardSession:
    """Başarılı ThingsBoard oturumunun bilgileri."""

    token: str
    refresh_token: Optional[str]
    user: ThingsBoardUser


class ThingsBoardAuthService:
    #tb rest api kullanılarak kullanıcı doğrular

    DEFAULT_USER_AGENT = "DocuMind-ThingsBoardAuthClient/1.0"

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
    ):
        configured_url = (
            base_url
            if base_url is not None
            else THINGSBOARD_URL
        )

        self.base_url = str(configured_url or "").strip().rstrip("/")

        if not self.base_url:
            raise ValueError("THINGSBOARD_URL tanımlanmamış.")

        parsed_url = urlparse(self.base_url)

        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError(
                "THINGSBOARD_URL http:// veya https:// ile başlamalıdır."
            )

        if not parsed_url.netloc:
            raise ValueError("THINGSBOARD_URL geçerli bir adres değil.")

        configured_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else THINGSBOARD_AUTH_TIMEOUT_SECONDS
        )

        try:
            self.timeout_seconds = float(configured_timeout)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "THINGSBOARD_AUTH_TIMEOUT_SECONDS sayısal olmalıdır."
            ) from error

        if self.timeout_seconds <= 0:
            raise ValueError(
                "ThingsBoard bağlantı zaman aşımı sıfırdan büyük olmalıdır."
            )

    def authenticate(
        self,
        email: str,
        password: str,
    ) -> ThingsBoardSession:
        """E-posta ve parola ile ThingsBoard kullanıcısını doğrular."""

        normalized_email = str(email or "").strip().lower()
        normalized_password = str(password or "")

        if not normalized_email or not normalized_password:
            raise ThingsBoardAuthError(
                "ThingsBoard e-posta adresi ve parola zorunludur."
            )

        login_response = self._request_json(
            path="/api/auth/login",
            method="POST",
            payload={
                # tb bu alanın adını username olarak bekler.
                "username": normalized_email,
                "password": normalized_password,
            },
            request_name="ThingsBoard giriş isteği",
        )

        token = str(login_response.get("token") or "").strip()

        if not token:
            raise ThingsBoardAuthError(
                "ThingsBoard giriş yanıtında oturum anahtarı bulunamadı."
            )

        refresh_token_value = login_response.get("refreshToken")
        refresh_token = (
            str(refresh_token_value).strip()
            if refresh_token_value
            else None
        )

        profile_response = self._request_json(
            path="/api/auth/user",
            method="GET",
            token=token,
            request_name="ThingsBoard kullanıcı bilgisi isteği",
        )

        user = self._to_user(
            profile_response,
            fallback_email=normalized_email,
        )

        return ThingsBoardSession(
            token=token,
            refresh_token=refresh_token,
            user=user,
        )

    def _request_json(
        self,
        path: str,
        method: str,
        payload: Optional[Dict[str, Any]] = None,
        token: Optional[str] = None,
        request_name: str = "ThingsBoard isteği",
    ) -> Dict[str, Any]:
        """ThingsBoard API'ye istek gönderir ve JSON yanıtını döndürür."""

        url = f"{self.base_url}/{path.lstrip('/')}"

        headers = {
            "Accept": "application/json",
            "User-Agent": self.DEFAULT_USER_AGENT,
        }

        request_body = None

        if payload is not None:
            headers["Content-Type"] = "application/json"
            request_body = json.dumps(
                payload,
                ensure_ascii=False,
            ).encode("utf-8")

        if token:
            # ThingsBoard REST API'nin kullandığı yetkilendirme başlığı.
            headers["X-Authorization"] = f"Bearer {token}"

        request = urllib.request.Request(
            url=url,
            data=request_body,
            headers=headers,
            method=method.upper(),
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                response_body = response.read().decode(
                    "utf-8",
                    errors="replace",
                )

        except urllib.error.HTTPError as error:
            self._raise_http_error(
                error=error,
                request_name=request_name,
            )

        except urllib.error.URLError as error:
            reason = getattr(error, "reason", None)

            if isinstance(reason, socket.timeout):
                raise ThingsBoardAuthError(
                    "ThingsBoard bağlantısı zaman aşımına uğradı."
                ) from error

            print(
                f"[ThingsBoard bağlantı hatası] {request_name}: {reason}"
            )

            raise ThingsBoardAuthError(
                "ThingsBoard sunucusuna bağlanılamadı."
            ) from error

        except (TimeoutError, socket.timeout) as error:
            raise ThingsBoardAuthError(
                "ThingsBoard bağlantısı zaman aşımına uğradı."
            ) from error

        if not response_body.strip():
            raise ThingsBoardAuthError(
                f"{request_name} boş yanıt döndürdü."
            )

        try:
            decoded_response = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise ThingsBoardAuthError(
                f"{request_name} geçerli bir JSON yanıtı döndürmedi."
            ) from error

        if not isinstance(decoded_response, dict):
            raise ThingsBoardAuthError(
                f"{request_name} beklenmeyen bir yanıt döndürdü."
            )

        return decoded_response

    def _raise_http_error(
        self,
        error: urllib.error.HTTPError,
        request_name: str,
    ) -> None:
       
        status_code = error.code
        try:
            raw_body = error.read().decode("utf-8", errors="replace")
        except Exception:
            raw_body = ""

        server_message = None
        if raw_body.strip():
            try:
                parsed_error = json.loads(raw_body)
                if isinstance(parsed_error, dict):
                    server_message = parsed_error.get("message")
            except json.JSONDecodeError:
                pass

        print(
            f"[ThingsBoard HTTP hata] {request_name} -> "
            f"HTTP {status_code} | mesaj: {server_message!r} | "
            f"ham gövde: {raw_body[:500]!r}"
        )

        if status_code == 401:
            raise ThingsBoardAuthError(
                "ThingsBoard kullanıcı adı veya parolası doğrulanamadı."
            ) from error

        if status_code == 403:
            raise ThingsBoardAuthError(
                "ThingsBoard hesabının bu işleme erişim yetkisi yok."
            ) from error

        if status_code == 404:
            raise ThingsBoardAuthError(
                "ThingsBoard kimlik doğrulama adresi bulunamadı. "
                "THINGSBOARD_URL değerini kontrol edin."
            ) from error

        if status_code == 429:
            raise ThingsBoardAuthError(
                "Çok fazla giriş denemesi yapıldı. "
                "Bir süre bekleyip yeniden deneyin."
            ) from error

        if status_code >= 500:
            raise ThingsBoardAuthError(
                "ThingsBoard sunucusu geçici olarak yanıt veremiyor."
            ) from error

        raise ThingsBoardAuthError(
            f"{request_name} başarısız oldu (HTTP {status_code})."
        ) from error

    @staticmethod
    def _entity_id(value: Any) -> Optional[str]:
        """ThingsBoard entity kimliği alanını metne dönüştürür."""

        if isinstance(value, dict):
            entity_id = value.get("id")

            if entity_id is not None:
                return str(entity_id)

            return None

        if value is None:
            return None

        return str(value)

    def _to_user(
        self,
        payload: Dict[str, Any],
        fallback_email: str,
    ) -> ThingsBoardUser:
        """ThingsBoard profil yanıtını kullanıcı nesnesine dönüştürür."""

        user_id = self._entity_id(payload.get("id"))

        if not user_id:
            raise ThingsBoardAuthError(
                "ThingsBoard kullanıcı profilinde kullanıcı kimliği bulunamadı."
            )

        email = str(
            payload.get("email")
            or fallback_email
        ).strip().lower()

        if not email:
            raise ThingsBoardAuthError(
                "ThingsBoard kullanıcı profilinde e-posta adresi bulunamadı."
            )

        return ThingsBoardUser(
            user_id=user_id,
            email=email,
            first_name=str(
                payload.get("firstName") or ""
            ).strip(),
            last_name=str(
                payload.get("lastName") or ""
            ).strip(),
            authority=str(
                payload.get("authority") or ""
            ).strip(),
            tenant_id=self._entity_id(payload.get("tenantId")),
            customer_id=self._entity_id(payload.get("customerId")), #kullanıcı giriş yaptığında döner
        )


def authenticate_thingsboard_user(
    email: str,
    password: str,
) -> ThingsBoardSession:
    """Varsayılan ayarlarla ThingsBoard kullanıcısını doğrular."""

    return ThingsBoardAuthService().authenticate(
        email=email,
        password=password,
    )