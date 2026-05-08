from .user import db


class Setting(db.Model):
    """
    Key-value store for admin-configurable application settings.
    Stored in the database so changes take effect without a redeploy.

    Current keys
    ------------
    ai_provider  : 'gemini' | 'openai' | 'none'  (default: 'gemini')
    """

    __tablename__ = 'settings'

    key = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.String(500), nullable=False, default='')

    @classmethod
    def get(cls, key: str, default: str = '') -> str:
        row = cls.query.get(key)
        return row.value if row else default

    @classmethod
    def set(cls, key: str, value: str) -> None:
        row = cls.query.get(key)
        if row:
            row.value = value
        else:
            db.session.add(cls(key=key, value=value))
        db.session.commit()
