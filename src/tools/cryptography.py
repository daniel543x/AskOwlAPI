from cryptography.fernet import Fernet

from .settings import settings


class CryptoService:
    """Symmetric encryption for secrets at rest (e.g. provider API keys)."""

    def __init__(self) -> None:
        self.token = Fernet(settings.MASTER_KEY.encode())

    def encrypt(self, text: str) -> str:
        return self.token.encrypt(text.encode()).decode()

    def decrypt(self, text: str) -> str:
        return self.token.decrypt(text.encode()).decode()


# Dependency
def get_crypto_service():
    return CryptoService()
