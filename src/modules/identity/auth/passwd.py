import bcrypt


def hash_passwd(password: str) -> str:
    """Bcrypt operates on bytes. Text -> Bytes -> Hash -> Text"""
    salt = bcrypt.gensalt(rounds=12)
    password_bytes = password.encode("utf-8")
    hashed_password_bytes = bcrypt.hashpw(password_bytes, salt)
    return hashed_password_bytes.decode("utf-8")


def check_passwd(plain_passwd: str, hashed_passwd: str) -> bool:
    """Bcrypt operates on bytes. Text -> Bytes -> Comparison"""
    plain_password_bytes = plain_passwd.encode("utf-8")
    hashed_password_bytes = hashed_passwd.encode("utf-8")
    return bcrypt.checkpw(plain_password_bytes, hashed_password_bytes)
