"""
ML Inference utility for dog breed identification.

Model strategy
--------------
We use EfficientNetB3 fine-tuned on the Stanford Dogs dataset (120 breeds)
instead of the original MobileNetV2-on-ImageNet approach.

MobileNetV2 + ImageNet had two critical flaws:
  1. It was never fine-tuned for dog breeds — ImageNet accuracy on dog
     breed discrimination is ~40-55%.
  2. decode_predictions() returns ImageNet synset labels (e.g.
     'n02085782_japanese_spaniel'), not clean breed names.

Fine-tuning instructions (run once, outside the web app):
  See scripts/train_breed_classifier.py for the full pipeline.
  After training, export to ONNX:
    torch.onnx.export(model, dummy_input, 'models/dog_breed_efficientnet.onnx')
  Place the .onnx file at:
    app/utils/models/dog_breed_efficientnet.onnx

Runtime fallback
----------------
If the fine-tuned ONNX model is not found (e.g. during initial development),
the module falls back to EfficientNetB3 pretrained on ImageNet with dog-class
filtering and clean label mapping.  This is more accurate than raw MobileNetV2
for the 120 Stanford Dogs breeds that overlap with ImageNet.
"""

import logging
import os
from typing import Callable

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

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

# Minimum confidence to report a result; below this we say "uncertain"
_CONFIDENCE_THRESHOLD = 0.35

# ---------------------------------------------------------------------------
# Model loader  (lazy-loaded singleton)
# ---------------------------------------------------------------------------
_predict_fn: Callable | None = None


def _load_model() -> Callable:
    """
    Load inference backend.  Priority:
      1. Fine-tuned ONNX model (app/utils/models/dog_breed_efficientnet.onnx)
      2. EfficientNetB3 pretrained on ImageNet (fallback, requires tensorflow)
    Returns a callable: (img_array: np.ndarray) -> (breed: str, confidence: float)
    """
    onnx_path = os.path.join(os.path.dirname(__file__), 'models', 'dog_breed_efficientnet.onnx')

    if os.path.exists(onnx_path):
        logger.info('Loading fine-tuned ONNX model from %s', onnx_path)
        return _load_onnx_model(onnx_path)

    logger.warning(
        'Fine-tuned ONNX model not found at %s. '
        'Falling back to ImageNet EfficientNetB3. '
        'Accuracy on non-common breeds will be limited. '
        'Run scripts/train_breed_classifier.py to produce the fine-tuned model.',
        onnx_path,
    )
    return _load_efficientnet_fallback()


def _load_onnx_model(onnx_path: str) -> Callable:
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx_path,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
        )
        input_name = sess.get_inputs()[0].name

        # Load the class label list saved alongside the model
        labels_path = onnx_path.replace('.onnx', '_labels.txt')
        with open(labels_path) as f:
            labels = [line.strip() for line in f]

        def _infer(img_array: np.ndarray) -> tuple[str, float]:
            outputs = sess.run(None, {input_name: img_array})
            probs = outputs[0][0]
            idx = int(np.argmax(probs))
            confidence = float(probs[idx])
            breed = labels[idx] if idx < len(labels) else 'Unknown'
            return breed, round(confidence * 100, 2)

        logger.info('ONNX model loaded successfully (%d classes).', len(labels))
        return _infer

    except Exception:
        logger.exception('Failed to load ONNX model, falling back to EfficientNetB3.')
        return _load_efficientnet_fallback()


def _load_efficientnet_fallback() -> Callable:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import (
        EfficientNetB3,
        preprocess_input,
        decode_predictions,
    )

    logger.info('Loading EfficientNetB3 (ImageNet weights) …')
    model = EfficientNetB3(weights='imagenet', include_top=True)
    # Warmup pass to avoid cold-start latency on first real request
    model.predict(np.zeros((1, 300, 300, 3)), verbose=0)
    logger.info('EfficientNetB3 loaded and warmed up.')

    def _infer(img_array: np.ndarray) -> tuple[str, float]:
        preds = model.predict(img_array, verbose=0)
        # top-5 so we can fall back to best dog class if top-1 is non-dog
        top5 = decode_predictions(preds, top=5)[0]

        # Find the highest-confidence prediction that maps to a dog breed
        best_breed: str | None = None
        best_conf: float = 0.0
        for _, synset_label, prob in top5:
            # ImageNet dog synsets start with 'n0208' range (class indices 151-268)
            # decode_predictions returns (synset_id, label, prob)
            # We match by label string → our clean map
            clean = _synset_to_clean(synset_label)
            if clean and float(prob) > best_conf:
                best_breed = clean
                best_conf = float(prob)

        if best_breed is None or best_conf < _CONFIDENCE_THRESHOLD:
            # Top-1 might still be a dog even if not in our map
            _, label, prob = top5[0]
            best_breed = label.replace('_', ' ').title()
            best_conf = float(prob)

        return best_breed, round(best_conf * 100, 2)

    return _infer


def _synset_to_clean(synset_label: str) -> str | None:
    """Map an ImageNet synset label string to a clean breed name."""
    normalised = synset_label.lower().replace('-', '_').replace(' ', '_')
    for clean in IMAGENET_DOG_CLASSES.values():
        if clean.lower().replace(' ', '_').replace('-', '_') == normalised:
            return clean
    return None


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
        Confidence percentage (0–100).  Values below ~35 indicate uncertainty.
    """
    global _predict_fn
    if _predict_fn is None:
        _predict_fn = _load_model()

    img = Image.open(img_path).convert('RGB')

    # Determine target size from backend
    target_size = (300, 300)  # EfficientNetB3 default; ONNX model may differ

    img_resized = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img_resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    breed, confidence = _predict_fn(img_array)
    return breed, confidence