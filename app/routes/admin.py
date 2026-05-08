import logging

from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import check_password_hash
from ..__init__ import limiter
from ..models.user import db, User
from ..models.setting import Setting
from ..config import Config
from ..utils.ai_advisor import provider_key_status

admin_bp = Blueprint('admin', __name__)
logger = logging.getLogger(__name__)


def _require_admin():
    """Return a redirect if the request is not from an authenticated admin."""
    if 'admin' not in session:
        return redirect(url_for('admin.admin_login'))
    return None


@admin_bp.route('/admin_login', methods=['GET', 'POST'])
@limiter.limit('10 per 15 minutes')
def admin_login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if (
            Config.ADMIN_PASSWORD_HASH
            and email == Config.ADMIN_EMAIL.lower()
            and check_password_hash(Config.ADMIN_PASSWORD_HASH, password)
        ):
            session.clear()
            session['admin'] = True
            logger.info('Admin login from IP %s', request.remote_addr)
            return redirect(url_for('admin.admin_dashboard'))

        flash('Invalid admin credentials.', 'error')
        logger.warning('Failed admin login from IP %s', request.remote_addr)

    return render_template('admin_login.html')


@admin_bp.route('/admin')
def admin_dashboard():
    guard = _require_admin()
    if guard:
        return guard

    users = User.query.order_by(User.created_at.desc()).all()
    return render_template('admin.html', users=users)


@admin_bp.route('/delete_user/<int:user_id>', methods=['POST'])
def delete_user(user_id):
    guard = _require_admin()
    if guard:
        return guard

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()

    logger.info('Admin deleted user id=%s', user_id)
    flash('User deleted.', 'success')
    return redirect(url_for('admin.admin_dashboard'))


@admin_bp.route('/edit_user/<int:user_id>', methods=['GET', 'POST'])
def edit_user(user_id):
    guard = _require_admin()
    if guard:
        return guard

    user = User.query.get_or_404(user_id)

    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        new_password = request.form.get('password') or ''

        if not username or not email:
            flash('Username and email are required.', 'error')
            return render_template('edit_user.html', user=user)

        user.username = username
        user.email = email
        if new_password:
            if len(new_password) < 8:
                flash('New password must be at least 8 characters.', 'error')
                return render_template('edit_user.html', user=user)
            user.set_password(new_password)

        db.session.commit()
        logger.info('Admin updated user id=%s', user_id)
        flash('User updated.', 'success')
        return redirect(url_for('admin.admin_dashboard'))

    return render_template('edit_user.html', user=user)


@admin_bp.route('/admin_logout')
def admin_logout():
    session.clear()
    logger.info('Admin logged out from IP %s', request.remote_addr)
    return redirect(url_for('auth.login'))


# ---------------------------------------------------------------------------
# AI Settings
# ---------------------------------------------------------------------------

_VALID_PROVIDERS = {'gemini', 'openai', 'none'}


@admin_bp.route('/admin/settings', methods=['GET', 'POST'])
def ai_settings():
    guard = _require_admin()
    if guard:
        return guard

    if request.method == 'POST':
        provider = (request.form.get('ai_provider') or 'none').strip().lower()
        if provider not in _VALID_PROVIDERS:
            flash('Invalid AI provider selected.', 'error')
        else:
            Setting.set('ai_provider', provider)
            logger.info('Admin changed AI provider to "%s" from IP %s', provider, request.remote_addr)
            flash(f'AI provider updated to "{provider}".', 'success')
        return redirect(url_for('admin.ai_settings'))

    current_provider = Setting.get('ai_provider', 'gemini')
    key_status = provider_key_status()
    return render_template(
        'admin_settings.html',
        current_provider=current_provider,
        key_status=key_status,
    )