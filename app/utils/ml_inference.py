"""
ML Inference utility for dog breed identification.

Model strategy (priority order)
--------------------------------
1. Fine-tuned ONNX model  (app/utils/models/dog_breed_efficientnet.onnx)
   Best accuracy. Train once with scripts/train_breed_classifier.py, export to
   ONNX, commit the file. No external dependencies at runtime.

2. HuggingFace Inference API  (free-tier fallback, no local GPU/RAM required)
   Model: google/vit-base-patch16-224 — trained on ImageNet which includes
   120 Stanford Dogs breed classes.  Accuracy is comparable to EfficientNetB3
   ImageNet weights and requires zero RAM on the server.
   Set HUGGINGFACE_API_KEY env var (free account at huggingface.co → Settings
   → Access Tokens) for higher rate limits (~1 000 req/day on free tier).
   Without a key the API still works but is rate-limited to ~10 req/min.

Note: TensorFlow has been removed.  It required ~1.5 GB RAM (incompatible
with free Railway / Render tiers) and produced a 1.1 GB Docker image.
Removal shrinks the image to ~250 MB and RAM to ~256 MB.
"""

import logging
import os
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# HuggingFace model used as the free-tier fallback (no local ML framework needed)
_HF_MODEL = 'google/vit-base-patch16-224'
_HF_API_URL = f'https://api-inference.huggingface.co/models/{_HF_MODEL}'

# ---------------------------------------------------------------------------
# ImageNet class index → clean breed name
# Covers the ~120 dog breed synsets that are in both ImageNet and Stanford Dogs.
# Keys are ImageNet class indices (0-999).
# ---------------------------------------------------------------------------
IMAGENET_DOG_CLASSES: dict[int, str] = {
    151: "Chihuahua", 152: "Japanese Spaniel", 153: "Maltese", 154: "Pekinese",
    155: "Shih-Tzu", 156: "Blenheim Spaniel", 157: "Papillon", 158: "Toy Terrier",
    159: "Rhodesian Ridgeback", 160: "Afghan Hound", 161: "Basset Hound",
    162: "Beagle", 163: "Bloodhound", 164: "Bluetick", 165: "Black-and-Tan Coonhound",
    166: "Walker Hound", 167: "English Foxhound", 168: "Redbone",
    169: "Borzoi", 170: "Irish Wolfhound", 171: "Italian Greyhound",
    172: "Whippet", 173: "Ibizan Hound", 174: "Norwegian Elkhound",
    175: "Otterhound", 176: "Saluki", 177: "Scottish Deerhound",
    178: "Weimaraner", 179: "Staffordshire Bull Terrier",
    180: "American Staffordshire Terrier", 181: "Bedlington Terrier",
    182: "Border Terrier", 183: "Kerry Blue Terrier", 184: "Irish Terrier",
    185: "Norfolk Terrier", 186: "Norwich Terrier", 187: "Yorkshire Terrier",
    188: "Wire Fox Terrier", 189: "Lakeland Terrier", 190: "Sealyham Terrier",
    191: "Airedale", 192: "Cairn Terrier", 193: "Australian Terrier",
    194: "Dandie Dinmont", 195: "Boston Bull", 196: "Miniature Schnauzer",
    197: "Giant Schnauzer", 198: "Standard Schnauzer", 199: "Scotch Terrier",
    200: "Tibetan Terrier", 201: "Silky Terrier", 202: "Soft-Coated Wheaten Terrier",
    203: "West Highland White Terrier", 204: "Lhasa Apso",
    205: "Flat-Coated Retriever", 206: "Curly-Coated Retriever",
    207: "Golden Retriever", 208: "Labrador Retriever",
    209: "Chesapeake Bay Retriever", 210: "German Short-Haired Pointer",
    211: "Vizsla", 212: "English Setter", 213: "Irish Setter",
    214: "Gordon Setter", 215: "Brittany Spaniel", 216: "Clumber Spaniel",
    217: "English Springer Spaniel", 218: "Welsh Springer Spaniel",
    219: "Cocker Spaniel", 220: "Sussex Spaniel", 221: "Irish Water Spaniel",
    222: "Kuvasz", 223: "Schipperke", 224: "Groenendael", 225: "Malinois",
    226: "Briard", 227: "Kelpie", 228: "Komondor", 229: "Old English Sheepdog",
    230: "Shetland Sheepdog", 231: "Collie", 232: "Border Collie",
    233: "Bouvier des Flandres", 234: "Rottweiler", 235: "German Shepherd",
    236: "Doberman", 237: "Miniature Pinscher", 238: "Greater Swiss Mountain Dog",
    239: "Bernese Mountain Dog", 240: "Appenzeller", 241: "Entlebucher",
    242: "Boxer", 243: "Bull Mastiff", 244: "Tibetan Mastiff",
    245: "French Bulldog", 246: "Great Dane", 247: "Saint Bernard",
    248: "Eskimo Dog", 249: "Malamute", 250: "Siberian Husky",
    251: "Affenpinscher", 252: "Basenji", 253: "Pug",
    254: "Leonberg", 255: "Newfoundland", 256: "Great Pyrenees",
    257: "Samoyed", 258: "Pomeranian", 259: "Chow Chow",
    260: "Keeshond", 261: "Brabancon Griffon", 262: "Pembroke Welsh Corgi",
    263: "Cardigan Welsh Corgi", 264: "Toy Poodle", 265: "Miniature Poodle",
    266: "Standard Poodle", 267: "Mexican Hairless",
    268: "Dingo", 269: "Dhole",
    270: "African Hunting Dog",
}

# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------

# Build a lowercase lookup from all our clean breed names so HF API labels
# can be matched regardless of spacing / capitalisation / underscores.
_BREED_LABEL_LOOKUP: dict[str, str] = {}
for _breed in IMAGENET_DOG_CLASSES.values():
    _key = _breed.lower()
    _BREED_LABEL_LOOKUP[_key] = _breed
    _BREED_LABEL_LOOKUP[_key.replace(' ', '_')] = _breed
    _BREED_LABEL_LOOKUP[_key.replace(' ', '-')] = _breed


def _match_breed_label(raw: str) -> str | None:
    """
    Try to map a raw HuggingFace/ImageNet label string to a clean breed name.
    Handles formats like 'golden retriever', 'golden_retriever',
    'n02099601 golden retriever' (synset prefix).
    """
    lower = raw.lower().strip()
    if lower in _BREED_LABEL_LOOKUP:
        return _BREED_LABEL_LOOKUP[lower]
    # Strip optional synset prefix, e.g. 'n02099601 golden retriever'
    if ' ' in lower:
        suffix = lower.split(' ', 1)[1]
        if suffix in _BREED_LABEL_LOOKUP:
            return _BREED_LABEL_LOOKUP[suffix]
    # Partial / substring match as last resort
    for key, val in _BREED_LABEL_LOOKUP.items():
        if key in lower or lower in key:
            return val
    return None


# ---------------------------------------------------------------------------
# Model loader  (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_predict_fn: Callable | None = None


def _load_model() -> Callable:
    """
    Load inference backend.  Priority:
      1. Fine-tuned ONNX model (app/utils/models/dog_breed_efficientnet.onnx)
      2. HuggingFace Inference API (free, no local RAM for ML)
    Returns a callable: (img_path: str) -> (breed: str, confidence: float)
    """
    onnx_path = os.path.join(os.path.dirname(__file__), 'models', 'dog_breed_efficientnet.onnx')

    if os.path.exists(onnx_path):
        logger.info('Loading fine-tuned ONNX model from %s', onnx_path)
        return _load_onnx_model(onnx_path)

    logger.info(
        'Fine-tuned ONNX model not found at %s. '
        'Using HuggingFace Inference API (%s) as fallback. '
        'Set HUGGINGFACE_API_KEY env var for higher rate limits.',
        onnx_path, _HF_MODEL,
    )
    return _load_huggingface_fallback()


def _load_onnx_model(onnx_path: str) -> Callable:
    """Load the fine-tuned ONNX model. Callable signature: (img_path) -> (breed, confidence)."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        input_name = sess.get_inputs()[0].name
        input_shape = sess.get_inputs()[0].shape  # e.g. [1, 300, 300, 3]
        h = input_shape[1] if isinstance(input_shape[1], int) else 300
        w = input_shape[2] if isinstance(input_shape[2], int) else 300

        labels_path = onnx_path.replace('.onnx', '_labels.txt')
        with open(labels_path) as f:
            labels = [line.strip() for line in f]

        def _infer(img_path: str) -> tuple[str, float]:
            img = Image.open(img_path).convert('RGB').resize((w, h), Image.LANCZOS)
            img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
            outputs = sess.run(None, {input_name: img_array})
            probs = outputs[0][0]
            idx = int(np.argmax(probs))
            breed = labels[idx] if idx < len(labels) else 'Unknown'
            return breed, round(float(probs[idx]) * 100, 2)

        logger.info('ONNX model loaded successfully (%d classes).', len(labels))
        return _infer

    except Exception:
        logger.exception('Failed to load ONNX model, falling back to HuggingFace API.')
        return _load_huggingface_fallback()


def _load_huggingface_fallback() -> Callable:
    """
    Use the HuggingFace Inference API for breed detection.

    Model : google/vit-base-patch16-224 (ImageNet — 120 dog breed classes)
    Free tier : ~1 000 requests/day with a free API key.
    Without key: works but rate-limited (~10 req/min) and model may need
                 a cold-start warm-up (returns HTTP 503 for ~20 s on first call).

    Get a free key at https://huggingface.co → Settings → Access Tokens
    and set HUGGINGFACE_API_KEY in your Railway environment variables.
    """
    import requests as _requests

    api_key = os.getenv('HUGGINGFACE_API_KEY', '')
    _headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
    logger.info('HuggingFace fallback ready (model=%s, key_set=%s)', _HF_MODEL, bool(api_key))

    def _infer(img_path: str) -> tuple[str, float]:
        try:
            with open(img_path, 'rb') as fh:
                data = fh.read()

            resp = _requests.post(_HF_API_URL, headers=_headers, data=data, timeout=40)

            if resp.status_code == 503:
                # Model is cold-starting on HuggingFace servers (takes ~20 s)
                logger.warning('HuggingFace model is warming up (503). Tell user to retry.')
                return 'Model is warming up — please submit again in 30 seconds.', 0.0

            if resp.status_code == 429:
                logger.warning('HuggingFace rate limit hit. Set HUGGINGFACE_API_KEY.')
                return 'Service temporarily busy — please try again shortly.', 0.0

            if resp.status_code != 200:
                logger.error('HuggingFace API error %s: %s', resp.status_code, resp.text[:200])
                return 'Breed detection unavailable — please try again.', 0.0

            results = resp.json()
            if not isinstance(results, list) or not results:
                return 'Unable to identify breed.', 0.0

            # Walk top-10 results and return the first one that maps to a dog breed
            for item in results[:10]:
                label = item.get('label', '')
                score = float(item.get('score', 0))
                clean = _match_breed_label(label)
                if clean:
                    return clean, round(score * 100, 2)

            # No dog breed found — return the top label cleaned up
            top_label = results[0].get('label', 'Unknown Breed')
            if ' ' in top_label:          # strip synset prefix if present
                top_label = top_label.split(' ', 1)[1]
            return top_label.replace('_', ' ').title(), round(float(results[0].get('score', 0)) * 100, 2)

        except Exception:
            logger.exception('HuggingFace inference call failed.')
            return 'Breed detection failed — please try again.', 0.0

    return _infer


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_breed(img_path: str) -> tuple[str, float]:
    """
    Identify the dog breed from an image file.

    Parameters
    ----------
    img_path : str
        Absolute path to a JPEG/PNG image (already validated and EXIF-stripped
        by the upload handler).

    Returns
    -------
    breed : str
        Human-readable breed name.
    confidence : float
        Confidence percentage (0–100).  0 means the result is a fallback message.
    """
    global _predict_fn
    if _predict_fn is None:
        _predict_fn = _load_model()

    return _predict_fn(img_path)