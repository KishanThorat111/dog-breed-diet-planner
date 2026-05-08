"""
Application entrypoint.

Development:
    python run.py

Production (gunicorn):
    gunicorn run:app -w 4 -k sync --bind 0.0.0.0:8000

Note: use sync workers (not uvicorn/gevent) because TensorFlow/ONNX Runtime
are not async-safe.  Each worker loads the model once; use 2–4 workers on a
typical server.  For GPU deployment, use 1 worker per GPU.
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    import os
    port = int(os.getenv('PORT', 5000))
    # debug=False even in dev — use FLASK_DEBUG=1 env var instead
    app.run(host='0.0.0.0', port=port, debug=os.getenv('FLASK_DEBUG', '0') == '1')
