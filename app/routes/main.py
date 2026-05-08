import io
import logging
import os
import datetime
import uuid
import tempfile

from flask import Blueprint, current_app, flash, jsonify, render_template, request, redirect, send_file, session, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from ..utils.inference_queue import enqueue_prediction, get_result
from ..diet_data import diet_plans, get_size_category, get_diet_plan

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Cloudinary (production) / local disk (dev) image storage
# ------------------------------------------------------------------
def _upload_image(pil_image: Image.Image, filename: str) -> tuple[str, str]:
    """
    Save a PIL image and return (filepath_for_inference, public_url_for_browser).

    In production (CLOUDINARY_URL set): uploads to Cloudinary, returns the
    secure URL as the public URL and a temp file path for the worker.

    In development: saves to UPLOAD_FOLDER, returns both as local paths.
    """
    cloudinary_url = os.getenv('CLOUDINARY_URL')

    if cloudinary_url:
        import cloudinary
        import cloudinary.uploader
        # cloudinary.config() is auto-configured from CLOUDINARY_URL env var
        buf = io.BytesIO()
        pil_image.save(buf, format='JPEG', quality=90)
        buf.seek(0)
        public_id = f"dogai/{uuid.uuid4().hex}"
        result = cloudinary.uploader.upload(
            buf,
            public_id=public_id,
            resource_type='image',
        )
        public_url = result['secure_url']
        # Write to a temp file so the worker thread can read it via filepath
        tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
        pil_image.save(tmp.name, format='JPEG', quality=90)
        logger.debug('Image uploaded to Cloudinary: %s', public_url)
        return tmp.name, public_url
    else:
        upload_dir = os.path.abspath(current_app.config['UPLOAD_FOLDER'])
        os.makedirs(upload_dir, exist_ok=True)
        filepath = os.path.join(upload_dir, filename)
        pil_image.save(filepath, format='JPEG', quality=90)
        public_url = f'static/uploads/{filename}'
        return filepath, public_url

# ------------------------------------------------------------------
# Allowed MIME magic bytes (first bytes of valid image formats)
# ------------------------------------------------------------------
# Note: WebP requires BOTH bytes 0-3 == RIFF and bytes 8-11 == WEBP.
# A bare RIFF check would also match WAV audio.
_ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'webp'}
_MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5 MB (enforced by Flask MAX_CONTENT_LENGTH too)


def _validate_image(file_storage) -> tuple[bool, str]:
    """
    Validate an uploaded FileStorage object.
    Returns (is_valid, error_message).
    Checks: extension, file size, and magic bytes (actual content type).
    """
    filename = file_storage.filename or ''
    if '.' not in filename:
        return False, 'File has no extension.'

    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in _ALLOWED_EXTENSIONS:
        return False, f'File type ".{ext}" is not allowed. Use JPG, PNG, or WebP.'

    # Read the full file into memory to check size and magic bytes
    file_bytes = file_storage.read()
    file_storage.seek(0)  # reset so callers can re-read

    if len(file_bytes) > _MAX_IMAGE_BYTES:
        return False, 'File exceeds the 5 MB size limit.'

    if len(file_bytes) < 12:
        return False, 'File is too small to be a valid image.'

    is_jpeg = file_bytes[:3] == b'\xff\xd8\xff'
    is_png  = file_bytes[:4] == b'\x89PNG'
    is_webp = file_bytes[:4] == b'RIFF' and file_bytes[8:12] == b'WEBP'

    if not (is_jpeg or is_png or is_webp):
        return False, 'File content does not match a supported image format.'

    return True, ''


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@main_bp.route('/index', methods=['GET', 'POST'])
def index():
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            flash('No file selected.', 'error')
            return render_template('index.html', user=session['user'])

        valid, err = _validate_image(file)
        if not valid:
            flash(err, 'error')
            return render_template('index.html', user=session['user'])

        # Prefix with a short UUID to prevent filename collisions between users
        safe_name = secure_filename(file.filename)
        unique_filename = f"{uuid.uuid4().hex[:8]}_{safe_name}"

        try:
            # Strip EXIF metadata before saving (privacy + security)
            image = Image.open(file)
            image = image.convert('RGB')   # drop EXIF, alpha channel, etc.
            filepath, public_url = _upload_image(image, unique_filename)
        except Exception:
            logger.exception('Failed to process uploaded image.')
            flash('Could not process the image. Please upload a valid photo.', 'error')
            return render_template('index.html', user=session['user'])

        # Enqueue async inference — does not block the HTTP thread
        job_id = enqueue_prediction(filepath, session['user'])
        session['pending_job_id'] = job_id
        # Store a browser-renderable URL for the result page
        session['pending_image'] = public_url

        return render_template('index.html', user=session['user'], job_id=job_id)

    return render_template('index.html', user=session['user'])


@main_bp.route('/prediction_status/<job_id>')
def prediction_status(job_id):
    """
    Polling endpoint for async inference results.
    Returns JSON: {status: 'pending'|'done'|'error', ...result fields}
    """
    if 'user' not in session:
        return jsonify(error='Unauthorised'), 401

    result = get_result(job_id)
    if result is None:
        return jsonify(status='pending')

    if result.get('status') == 'error':
        return jsonify(status='error', message='Inference failed. Please try again.')

    breed = result['breed']
    diet = result['diet']
    confidence = result['confidence']
    size_category = result['size']

    youtube_link = (
        f"https://www.youtube.com/results?search_query="
        f"{breed.replace(' ', '+')}+dog+training+exercise"
    )
    amazon_link = (
        f"https://www.amazon.in/s?k="
        f"{diet['food'].replace(' ', '+')}"
    )

    # Store in session for the download report
    session['breed'] = breed
    session['confidence'] = confidence
    session['size'] = size_category
    session['diet'] = diet

    logger.info('Prediction ready: user=%s breed=%s confidence=%.2f', session['user'], breed, confidence)

    return jsonify(
        status='done',
        breed=breed,
        confidence=confidence,
        size=size_category,
        diet=diet,
        image=session.get('pending_image', ''),
        youtube_link=youtube_link,
        amazon_link=amazon_link,
        ai_advice=result.get('ai_advice'),
        ai_provider=result.get('ai_provider', 'none'),
    )


@main_bp.route('/download_report')
def download_report():
    if 'user' not in session:
        return redirect(url_for('auth.login'))

    breed = session.get('breed', 'Unknown')
    confidence = session.get('confidence', 0)
    size = session.get('size', 'Unknown')
    diet = session.get('diet', {})

    report_text = (
        "Dog Identification & Diet Report\n"
        "---------------------------------\n\n"
        f"Breed         : {breed}\n"
        f"Confidence    : {confidence}%\n"
        f"Size Category : {size}\n\n"
        "Recommended Diet Plan\n"
        "---------------------\n"
        f"Main Food     : {diet.get('food', 'N/A')}\n"
        f"Meals Per Day : {diet.get('meals', 'N/A')}\n"
        f"Extras        : {diet.get('extras', 'N/A')}\n\n"
        f"Generated on  : {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n"
    )

    buf = io.BytesIO(report_text.encode('utf-8'))
    buf.seek(0)
    return send_file(
        buf,
        as_attachment=True,
        download_name='dog_diet_report.txt',
        mimetype='text/plain',
    )


@main_bp.route('/about')
def about():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('about.html', user=session['user'])


@main_bp.route('/services')
def services():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('services.html', user=session['user'])


@main_bp.route('/vets')
def vets():
    if 'user' not in session:
        return redirect(url_for('auth.login'))
    return render_template('vets.html', user=session['user'])