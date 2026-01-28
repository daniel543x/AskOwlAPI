# https://cryptography.io/en/latest/fernet/
import os

from cryptography.fernet import Fernet

try:
    MASTER_KEY = os.environ["MASTER_KEY"]
except:
    raise ValueError("Var MASTER_KEY is none!")

try:
    token = Fernet(MASTER_KEY)
except (ValueError, TypeError):
    raise ValueError("MASTER_KEY has invalid format. Error: {e}")

token = Fernet(MASTER_KEY.encode())


def encrypt(text: str) -> str:
    return token.encrypt(text.encode()).decode()


def decrypt(encrypt_text: str) -> str:
    return token.decrypt(encrypt_text.encode()).decode()
