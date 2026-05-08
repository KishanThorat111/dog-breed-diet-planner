import logging
import re

from flask import Blueprint, render_template, request, redirect, url_for, session, flash
from werkzeug.security import check_password_hash
from ..__init__ import limiter
from ..models.user import db, User
from ..config import Config

auth_bp = Blueprint('auth', __name__)
logger = logging.getLogger(__name__)

_EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


@auth_bp.route('/')
def home_redirect():
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
@limiter.limit('5 per hour')
def register():
    if request.method == 'POST':
        username = (request.form.get('username') or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        # --- Input validation ---
        errors = []
        if not username or len(username) < 2 or len(username) > 50:
            errors.append('Username must be between 2 and 50 characters.')
        if not re.match(r'^[A-Za-z0-9_.-]+$', username):
            errors.append('Username may only contain letters, numbers, underscores, hyphens, and dots.')
        if not email or not _EMAIL_RE.match(email):
            errors.append('A valid email address is required.')
        if len(password) < 8:
            errors.append('Password must be at least 8 characters.')

        if errors:
            for e in errors:
                flash(e, 'error')
            return render_template('register.html')

        if User.query.filter_by(email=email).first():
            flash('An account with that email already exists.', 'error')
            return render_template('register.html')

        if User.query.filter_by(username=username).first():
            flash('That username is already taken.', 'error')
            return render_template('register.html')

        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()

        logger.info('New user registered: %s', email)
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('auth.login'))

    return render_template('register.html')


@auth_bp.route('/login', methods=['GET', 'POST'])
@limiter.limit('10 per 15 minutes')
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        password = request.form.get('password') or ''

        if not email or not password:
            flash('Email and password are required.', 'error')
            return render_template('login.html')

        # Admin login
        if (
            Config.ADMIN_PASSWORD_HASH
            and email == Config.ADMIN_EMAIL.lower()
            and check_password_hash(Config.ADMIN_PASSWORD_HASH, password)
        ):
            session.clear()
            session['admin'] = True
            logger.info('Admin login from IP %s', request.remote_addr)
            return redirect(url_for('admin.admin_dashboard'))

        # Normal user login
        user = User.query.filter_by(email=email).first()
        if user and user.is_active and user.check_password(password):
            session.clear()
            session['user'] = user.username
            logger.info('User login: %s', email)
            return redirect(url_for('main.index'))

        # Generic message — do not reveal whether email exists
        flash('Invalid email or password.', 'error')
        logger.warning('Failed login attempt for email: %s from IP %s', email, request.remote_addr)

    return render_template('login.html')


@auth_bp.route('/logout')
def logout():
    username = session.get('user', 'unknown')
    session.clear()
    logger.info('User logged out: %s', username)
    return redirect(url_for('auth.login'))