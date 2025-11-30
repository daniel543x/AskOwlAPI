import asyncio

import bcrypt


async def hash_passwd(password: str) -> str:
    """Bcrypt operates on bytes. Text -> Bytes -> Hash -> Text.
    Runs in a separate thread to avoid blocking the event loop."""

    def _hash():
        salt = bcrypt.gensalt(rounds=12)
        password_bytes = password.encode("utf-8")
        hashed_password_bytes = bcrypt.hashpw(password_bytes, salt)
        return hashed_password_bytes.decode("utf-8")

    return await asyncio.to_thread(_hash)


async def check_passwd(plain_passwd: str, hashed_passwd: str) -> bool:
    """Bcrypt operates on bytes. Text -> Bytes -> Comparison.
    Runs in a separate thread to avoid blocking the event loop."""

    def _check():
        plain_password_bytes = plain_passwd.encode("utf-8")
        hashed_password_bytes = hashed_passwd.encode("utf-8")
        return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)

    return await asyncio.to_thread(_check)
