import logging
import os
from flask import Flask, jsonify
from flask_wtf.csrf import CSRFProtect
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_talisman import Talisman
from .config import Config
from .models.user import db, User
from .models.setting import Setting  # noqa: F401 — imported so db.create_all() includes it

# Module-level logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger(__name__)

csrf = CSRFProtect()
limiter = Limiter(key_func=get_remote_address)


def create_app() -> Flask:
    # Validate critical config before creating the app
    Config.validate()

    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.config.from_object(Config)

    # Ensure upload folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    # Extensions
    db.init_app(app)
    csrf.init_app(app)
    limiter.init_app(app)

    # Security headers (relaxed CSP to allow Bootstrap CDN, Google Fonts, Cloudinary)
    is_production = not app.config.get('DEBUG', False)
    Talisman(
        app,
        force_https=is_production,
        strict_transport_security=is_production,
        content_security_policy={
            'default-src': ["'self'"],
            'style-src': [
                "'self'",
                "'unsafe-inline'",
                'https://cdn.jsdelivr.net',
                'https://fonts.googleapis.com',
            ],
            'font-src': ["'self'", 'https://fonts.gstatic.com'],
            'script-src': ["'self'", 'https://cdn.jsdelivr.net'],
            'img-src': [
                "'self'",
                'data:',
                'https://res.cloudinary.com',  # Cloudinary-hosted images
            ],
        },
        frame_options='DENY',
        referrer_policy='strict-origin-when-cross-origin',
    )

    with app.app_context():
        db.create_all()
        # Pass the app so the worker thread can push an app context for DB access
        from .utils.inference_queue import start_worker_thread
        start_worker_thread(app)

    # Register blueprints
    from .routes.auth import auth_bp
    from .routes.main import main_bp
    from .routes.admin import admin_bp

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(admin_bp)

    # Global error handlers
    @app.errorhandler(404)
    def not_found(e):
        return jsonify(error='Not found'), 404

    @app.errorhandler(413)
    def file_too_large(e):
        return jsonify(error='File too large. Maximum size is 5 MB.'), 413

    @app.errorhandler(429)
    def ratelimit_handler(e):
        return jsonify(error=f'Too many requests. {e.description}'), 429

    @app.errorhandler(500)
    def internal_error(e):
        logger.exception('Internal server error')
        return jsonify(error='An internal error occurred. Please try again.'), 500

    # Health check endpoint (used by Railway and load balancers)
    @app.route('/health')
    def health():
        return jsonify(status='ok'), 200

    logger.info('Application started successfully.')
    return app