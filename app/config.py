import os
import secrets
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Secret key: must be set via environment in production.
    # In development a random key is generated each run (sessions won't persist across restarts).
    _raw_secret = os.getenv('SECRET_KEY')
    SECRET_KEY: str = _raw_secret if _raw_secret else secrets.token_hex(32)

    # Railway (and Heroku) PostgreSQL URLs use the legacy 'postgres://' prefix
    # which SQLAlchemy 1.4+ rejects. Normalise to 'postgresql://' automatically.
    _db_url = os.getenv('DATABASE_URL', 'sqlite:///app.db')
    SQLALCHEMY_DATABASE_URI: str = (
        _db_url.replace('postgres://', 'postgresql://', 1)
        if _db_url.startswith('postgres://')
        else _db_url
    )
    SQLALCHEMY_TRACK_MODIFICATIONS: bool = False

    UPLOAD_FOLDER: str = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'static', 'uploads'
    )
    MAX_CONTENT_LENGTH: int = 5 * 1024 * 1024  # 5 MB hard limit on uploads

    ADMIN_EMAIL: str = os.getenv('ADMIN_EMAIL', 'admin@example.com')
    # Must be pre-hashed with werkzeug.security.generate_password_hash and stored in .env
    ADMIN_PASSWORD_HASH: str | None = os.getenv('ADMIN_PASSWORD_HASH')

    DEBUG: bool = os.getenv('DEBUG', 'False').lower() == 'true'

    @classmethod
    def validate(cls) -> None:
        """Fail fast at startup if critical configuration is missing."""
        errors = []
        if not os.getenv('SECRET_KEY'):
            import logging
            logging.getLogger(__name__).warning(
                "SECRET_KEY not set — using a random key. "
                "Sessions will be invalidated on every restart. "
                "Set SECRET_KEY in your .env file for persistent sessions."
            )
        if not cls.ADMIN_PASSWORD_HASH:
            errors.append(
                "ADMIN_PASSWORD_HASH is not set. "
                "Generate one with: python -c \"from werkzeug.security import "
                "generate_password_hash; print(generate_password_hash('yourpassword'))\""
            )
        if errors:
            raise RuntimeError(
                "Application cannot start due to missing configuration:\n" +
                "\n".join(f"  - {e}" for e in errors)
            )