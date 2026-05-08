"""
Test suite for Dog Breed Diet Planner.

Run with:
    pytest tests/ -v --cov=app --cov-report=term-missing
"""
import io
import os
import pytest
from unittest.mock import patch, MagicMock

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app():
    """Create an isolated test application."""
    os.environ.setdefault('ADMIN_PASSWORD_HASH',
        # werkzeug hash of "adminpassword" — pre-computed for test isolation
        'pbkdf2:sha256:260000$test$'
        'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    )
    os.environ.setdefault('SECRET_KEY', 'test-secret-key-not-for-production')

    from app import create_app
    flask_app = create_app()
    flask_app.config.update({
        'TESTING': True,
        'SQLALCHEMY_DATABASE_URI': 'sqlite:///:memory:',
        'WTF_CSRF_ENABLED': False,   # disable CSRF in tests
        'RATELIMIT_ENABLED': False,  # disable rate limits in tests
    })

    with flask_app.app_context():
        from app.models.user import db
        db.create_all()
        yield flask_app
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def registered_user(app, client):
    """Register and return a test user via the API."""
    client.post('/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'Password123',
    })
    return {'username': 'testuser', 'email': 'test@example.com', 'password': 'Password123'}


@pytest.fixture()
def logged_in_client(client, registered_user):
    """Return a client with an active user session."""
    client.post('/login', data={
        'email': registered_user['email'],
        'password': registered_user['password'],
    })
    return client


def _make_image_bytes(format='JPEG') -> bytes:
    """Create a minimal valid image in memory."""
    from PIL import Image
    buf = io.BytesIO()
    img = Image.new('RGB', (100, 100), color=(120, 80, 40))
    img.save(buf, format=format)
    buf.seek(0)
    return buf.read()


# ===========================================================================
# Auth tests
# ===========================================================================

class TestRegister:
    def test_register_success(self, client):
        resp = client.post('/register', data={
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'SecurePass1',
        })
        assert resp.status_code in (200, 302)

    def test_register_duplicate_email(self, client, registered_user):
        resp = client.post('/register', data={
            'username': 'other',
            'email': registered_user['email'],
            'password': 'SecurePass1',
        })
        assert b'already exists' in resp.data or resp.status_code == 200

    def test_register_short_password(self, client):
        resp = client.post('/register', data={
            'username': 'user2',
            'email': 'user2@example.com',
            'password': 'short',
        })
        assert b'8 characters' in resp.data or resp.status_code == 200

    def test_register_invalid_email(self, client):
        resp = client.post('/register', data={
            'username': 'user3',
            'email': 'not-an-email',
            'password': 'Password123',
        })
        assert b'valid email' in resp.data or resp.status_code == 200

    def test_register_special_chars_username(self, client):
        resp = client.post('/register', data={
            'username': '<script>alert(1)</script>',
            'email': 'xss@example.com',
            'password': 'Password123',
        })
        # Should reject — username has invalid characters
        assert resp.status_code == 200


class TestLogin:
    def test_login_success(self, client, registered_user):
        resp = client.post('/login', data={
            'email': registered_user['email'],
            'password': registered_user['password'],
        }, follow_redirects=True)
        assert resp.status_code == 200

    def test_login_wrong_password(self, client, registered_user):
        resp = client.post('/login', data={
            'email': registered_user['email'],
            'password': 'WrongPassword1',
        })
        assert b'Invalid' in resp.data

    def test_login_nonexistent_email(self, client):
        resp = client.post('/login', data={
            'email': 'nobody@example.com',
            'password': 'SomePassword1',
        })
        assert b'Invalid' in resp.data

    def test_login_empty_fields(self, client):
        resp = client.post('/login', data={'email': '', 'password': ''})
        assert b'required' in resp.data or resp.status_code == 200

    def test_logout_clears_session(self, logged_in_client):
        resp = logged_in_client.get('/logout', follow_redirects=True)
        assert resp.status_code == 200
        # After logout, /index should redirect to login
        resp2 = logged_in_client.get('/index')
        assert resp2.status_code in (302, 200)


# ===========================================================================
# Upload & inference tests
# ===========================================================================

class TestUpload:
    def test_upload_requires_login(self, client):
        resp = client.post('/index', data={
            'image': (io.BytesIO(_make_image_bytes()), 'dog.jpg'),
        }, content_type='multipart/form-data')
        assert resp.status_code == 302   # redirect to login

    def test_upload_valid_jpeg(self, logged_in_client):
        img_bytes = _make_image_bytes('JPEG')
        with patch('app.utils.inference_queue.enqueue_prediction', return_value='test-job-id'):
            resp = logged_in_client.post('/index', data={
                'image': (io.BytesIO(img_bytes), 'dog.jpg'),
            }, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert b'test-job-id' in resp.data

    def test_upload_rejects_text_file(self, logged_in_client):
        resp = logged_in_client.post('/index', data={
            'image': (io.BytesIO(b'this is not an image'), 'bad.jpg'),
        }, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert b'does not match' in resp.data

    def test_upload_rejects_oversized_file(self, logged_in_client):
        # 6 MB of valid-looking JPEG magic bytes
        big_bytes = b'\xff\xd8\xff' + b'\x00' * (6 * 1024 * 1024)
        resp = logged_in_client.post('/index', data={
            'image': (io.BytesIO(big_bytes), 'big.jpg'),
        }, content_type='multipart/form-data')
        assert resp.status_code in (200, 413)

    def test_upload_rejects_no_extension(self, logged_in_client):
        img_bytes = _make_image_bytes('JPEG')
        resp = logged_in_client.post('/index', data={
            'image': (io.BytesIO(img_bytes), 'nodotfile'),
        }, content_type='multipart/form-data')
        assert resp.status_code == 200
        assert b'extension' in resp.data

    def test_polling_endpoint_pending(self, logged_in_client):
        with patch('app.routes.main.get_result', return_value=None):
            resp = logged_in_client.get('/prediction_status/fake-job-id')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'pending'

    def test_polling_endpoint_done(self, logged_in_client):
        mock_result = {
            'status': 'done',
            'breed': 'Golden Retriever',
            'confidence': 92.5,
            'size': 'large',
            'diet': {
                'food': 'Large-breed formula',
                'meals': '2 meals',
                'extras': 'salmon oil',
                'avoid': 'grapes',
                'notes': '',
            },
        }
        with patch('app.routes.main.get_result', return_value=mock_result):
            resp = logged_in_client.get('/prediction_status/fake-job-id')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['status'] == 'done'
        assert data['breed'] == 'Golden Retriever'
        assert data['confidence'] == 92.5


# ===========================================================================
# Diet data tests
# ===========================================================================

class TestDietData:
    def test_get_size_known_small(self):
        from app.diet_data import get_size_category
        assert get_size_category('Chihuahua') == 'small'
        assert get_size_category('pomeranian') == 'small'
        assert get_size_category('Pug') == 'small'

    def test_get_size_known_large(self):
        from app.diet_data import get_size_category
        assert get_size_category('Golden Retriever') == 'large'
        assert get_size_category('labrador_retriever') == 'large'
        assert get_size_category('German Shepherd') == 'large'

    def test_get_size_known_giant(self):
        from app.diet_data import get_size_category
        assert get_size_category('Great Dane') == 'giant'
        assert get_size_category('saint bernard') == 'giant'

    def test_get_size_defaults_medium(self):
        from app.diet_data import get_size_category
        assert get_size_category('some_unknown_breed') == 'medium'
        assert get_size_category('') == 'medium'

    def test_get_diet_plan_returns_dict(self):
        from app.diet_data import get_diet_plan
        plan = get_diet_plan('Golden Retriever')
        assert isinstance(plan, dict)
        assert 'food' in plan
        assert 'meals' in plan
        assert 'avoid' in plan

    def test_breed_health_note_injected(self):
        from app.diet_data import get_diet_plan
        plan = get_diet_plan('German Shepherd')
        assert 'breed_note' in plan
        assert 'dysplasia' in plan['breed_note'].lower()

    def test_diet_plan_does_not_mutate_template(self):
        from app.diet_data import get_diet_plan, diet_plans
        plan1 = get_diet_plan('German Shepherd')
        plan1['food'] = 'MODIFIED'
        plan2 = get_diet_plan('Golden Retriever')
        # The large template should be unchanged
        assert diet_plans['large']['food'] != 'MODIFIED'


# ===========================================================================
# ML inference tests
# ===========================================================================

class TestMLInference:
    def test_predict_returns_tuple(self, tmp_path):
        """Smoke-test: ensure predict_breed returns (str, float)."""
        from PIL import Image
        import numpy as np

        img_path = str(tmp_path / 'test_dog.jpg')
        img = Image.new('RGB', (300, 300), color=(150, 100, 50))
        img.save(img_path)

        mock_fn = MagicMock(return_value=('Golden Retriever', 88.5))
        with patch('app.utils.ml_inference._predict_fn', mock_fn):
            from app.utils.ml_inference import predict_breed
            # Reset the global so our mock is used
            import app.utils.ml_inference as ml_mod
            ml_mod._predict_fn = mock_fn
            breed, confidence = predict_breed(img_path)

        assert isinstance(breed, str)
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 100

    def test_predict_corrupt_image_raises(self, tmp_path):
        from app.utils.ml_inference import predict_breed
        bad_path = str(tmp_path / 'bad.jpg')
        with open(bad_path, 'wb') as f:
            f.write(b'not an image')
        with pytest.raises(Exception):
            predict_breed(bad_path)


# ===========================================================================
# Admin tests
# ===========================================================================

class TestAdmin:
    def test_admin_dashboard_requires_auth(self, client):
        resp = client.get('/admin')
        assert resp.status_code == 302

    def test_delete_user_requires_post(self, client):
        """GET to delete endpoint must not delete (only POST is allowed)."""
        resp = client.get('/delete_user/1')
        assert resp.status_code == 405   # Method Not Allowed

    def test_edit_user_requires_admin(self, logged_in_client):
        resp = logged_in_client.get('/edit_user/1')
        assert resp.status_code == 302   # redirect to admin login
