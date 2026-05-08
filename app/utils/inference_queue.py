"""
Async inference queue for dog breed prediction.

Architecture
------------
Submitting a prediction (POST /index) no longer blocks the HTTP thread.
Instead, the route:
  1. Saves the image, creates a job record (status=pending).
  2. Enqueues the job to Redis (or the in-process fallback queue).
  3. Returns immediately with a job_id.

A background worker thread picks up the job, runs inference, and stores the
result back into the job store so the result page can poll for it.

Redis usage
-----------
If REDIS_URL is set in the environment, the real Redis queue is used:
  - Queue: Redis List  `dogai:inference:queue`
  - Result: Redis Hash `dogai:inference:result:{job_id}`  (TTL 1 hour)

In-process fallback
-------------------
If Redis is not available (local dev without Docker), a Python queue.Queue
and a daemon thread are used instead.  Results are stored in a dict.
This mode does NOT survive worker restarts — fine for dev only.
"""

import json
import logging
import os
import queue
import threading
import uuid

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# In-process fallback storage
# ---------------------------------------------------------------------------
# WARNING: _local_results is per-process. Under multi-worker Gunicorn (without
# Redis), a job enqueued by worker A will be processed and stored in worker A's
# memory, but the polling request may land on worker B where the result is
# absent.  For production multi-worker deployments, set REDIS_URL.
_local_queue: queue.Queue = queue.Queue()
_local_results: dict[str, dict] = {}
_worker_started: bool = False
_worker_lock: threading.Lock = threading.Lock()

# Flask app reference — set by start_worker_thread() so the worker thread can
# push an app context when it needs to query the database (e.g. Setting.get).
_flask_app = None

# ---------------------------------------------------------------------------
# Redis client (optional)
# ---------------------------------------------------------------------------
_redis_client = None
_REDIS_QUEUE_KEY = 'dogai:inference:queue'
_REDIS_RESULT_TTL = 3600  # 1 hour


def _get_redis():
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    redis_url = os.getenv('REDIS_URL')
    if not redis_url:
        return None

    try:
        import redis
        _redis_client = redis.from_url(redis_url, decode_responses=True)
        _redis_client.ping()
        logger.info('Redis queue connected: %s', redis_url)
    except Exception:
        logger.warning('Redis not available — using in-process fallback queue.')
        _redis_client = None

    return _redis_client


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enqueue_prediction(img_path: str, user: str) -> str:
    """
    Add an inference job to the queue.
    Returns a job_id that can be used to poll for the result.
    """
    job_id = str(uuid.uuid4())
    payload = json.dumps({'job_id': job_id, 'img_path': img_path, 'user': user})

    r = _get_redis()
    if r:
        r.rpush(_REDIS_QUEUE_KEY, payload)
        logger.debug('Enqueued job %s to Redis.', job_id)
    else:
        _local_queue.put(payload)
        logger.debug('Enqueued job %s to local queue.', job_id)

    return job_id


def get_result(job_id: str) -> dict | None:
    """
    Return the inference result for a job_id, or None if not ready yet.
    Result dict keys: breed, confidence, size, diet, status ('done'|'error')
    """
    r = _get_redis()
    if r:
        raw = r.hget(f'dogai:inference:result:{job_id}', 'data')
        return json.loads(raw) if raw else None

    return _local_results.get(job_id)


def _store_result(job_id: str, result: dict) -> None:
    r = _get_redis()
    if r:
        r.hset(f'dogai:inference:result:{job_id}', 'data', json.dumps(result))
        r.expire(f'dogai:inference:result:{job_id}', _REDIS_RESULT_TTL)
    else:
        _local_results[job_id] = result


def _process_job(payload_str: str) -> None:
    """Run inference for one job and store the result."""
    from .ml_inference import predict_breed
    from .ai_advisor import get_ai_advice
    from ..diet_data import get_size_category, get_diet_plan

    try:
        payload = json.loads(payload_str)
        job_id = payload['job_id']
        img_path = payload['img_path']

        logger.info('Processing inference job %s for user=%s', job_id, payload.get('user'))

        breed, confidence = predict_breed(img_path)
        size = get_size_category(breed)
        diet = get_diet_plan(breed)

        # Read AI provider from DB settings (needs app context)
        provider = 'gemini'  # safe default if no app context available
        if _flask_app:
            with _flask_app.app_context():
                from ..models.setting import Setting
                provider = Setting.get('ai_provider', 'gemini')

        ai_advice = get_ai_advice(breed, size, diet, provider)

        _store_result(job_id, {
            'status': 'done',
            'breed': breed,
            'confidence': confidence,
            'size': size,
            'diet': diet,
            'ai_advice': ai_advice,
            'ai_provider': provider,
        })
        logger.info('Inference job %s complete: breed=%s confidence=%.2f ai=%s',
                    job_id, breed, confidence, provider)

    except Exception:
        logger.exception('Inference job failed: %s', payload_str[:200])
        try:
            _store_result(payload['job_id'], {'status': 'error'})
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Background worker thread (in-process fallback)
# ---------------------------------------------------------------------------

def _worker_loop() -> None:
    logger.info('Inference worker thread started.')
    while True:
        try:
            payload_str = _local_queue.get(timeout=1)
            # Push app context so DB queries inside _process_job work
            if _flask_app:
                with _flask_app.app_context():
                    _process_job(payload_str)
            else:
                _process_job(payload_str)
            _local_queue.task_done()
        except queue.Empty:
            pass
        except Exception:
            logger.exception('Unexpected error in inference worker loop.')


def start_worker_thread(app=None) -> None:
    """
    Start the in-process background worker thread.
    Call once from create_app() when Redis is NOT available.
    Safe to call multiple times — only one thread is started.

    Pass the Flask `app` instance so the worker thread can push an app
    context when accessing the database (e.g. reading AI provider setting).
    """
    global _worker_started, _flask_app
    _flask_app = app

    if _get_redis():
        logger.info('Redis available — in-process worker thread not started.')
        return

    with _worker_lock:
        if _worker_started:
            logger.debug('Inference worker thread already running — skipping.')
            return
        _worker_started = True

    t = threading.Thread(target=_worker_loop, name='inference-worker', daemon=True)
    t.start()
    logger.info('In-process inference worker thread started (PID=%s).', os.getpid())
