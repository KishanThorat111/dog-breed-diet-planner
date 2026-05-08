# app/diet_data.py
# Per-breed and per-size-category diet plans.
# Calorie estimates are for an average adult dog at maintenance weight.
# Always consult a vet for dogs with health conditions.

# ---------------------------------------------------------------------------
# Size-category plans (fallback when breed not in BREED_OVERRIDES)
# ---------------------------------------------------------------------------
diet_plans: dict[str, dict] = {
    "small": {
        "food": "High-protein small-breed dry kibble",
        "meals": "3 meals per day",
        "calories_min": 200,
        "calories_max": 400,
        "extras": "Boiled chicken (no bones), baby carrots, blueberries",
        "avoid": "Grapes, raisins, onions, chocolate, xylitol",
        "notes": "Small breeds have fast metabolisms; never skip meals to prevent hypoglycaemia.",
    },
    "medium": {
        "food": "Balanced adult dry dog food (chicken or lamb base)",
        "meals": "2 meals per day",
        "calories_min": 400,
        "calories_max": 900,
        "extras": "Brown rice, scrambled eggs, cooked pumpkin",
        "avoid": "Avocado, macadamia nuts, raw onion, caffeine",
        "notes": "Adjust portions based on activity level.",
    },
    "large": {
        "food": "Large-breed joint-support formula (glucosamine added)",
        "meals": "2 meals per day",
        "calories_min": 900,
        "calories_max": 2100,
        "extras": "Salmon oil (omega-3), steamed broccoli, boiled lean beef",
        "avoid": "High-fat human food, cooked bones, alcohol, grapes",
        "notes": (
            "Large breeds are prone to bloat (GDV). "
            "Feed from a raised bowl and avoid vigorous exercise within 1 hour of meals."
        ),
    },
    "giant": {
        "food": "Giant-breed formula with joint and hip support",
        "meals": "2–3 smaller meals per day (reduces bloat risk)",
        "calories_min": 2100,
        "calories_max": 3500,
        "extras": "Fish oil, plain yoghurt, sweet potato",
        "avoid": "Rapid feeding, cooked bones, high-sodium food",
        "notes": "Giant breeds age faster; transition to senior food at ~5 years.",
    },
}

# ---------------------------------------------------------------------------
# Breed-specific overrides
# Format: breed_name_lower_normalised -> {size, extra notes, calories_min/max}
# ---------------------------------------------------------------------------
BREED_SIZE_MAP: dict[str, str] = {
    # ---- Small ----
    "chihuahua": "small",
    "japanese spaniel": "small",
    "maltese": "small",
    "pekinese": "small",
    "shih-tzu": "small",
    "shih tzu": "small",
    "blenheim spaniel": "small",
    "papillon": "small",
    "toy terrier": "small",
    "pomeranian": "small",
    "yorkshire terrier": "small",
    "norfolk terrier": "small",
    "norwich terrier": "small",
    "cairn terrier": "small",
    "scottish terrier": "small",
    "scotch terrier": "small",
    "silky terrier": "small",
    "toy poodle": "small",
    "miniature pinscher": "small",
    "affenpinscher": "small",
    "brabancon griffon": "small",
    "lhasa apso": "small",
    "tibetan terrier": "small",
    "west highland white terrier": "small",
    "dandie dinmont": "small",
    "miniature schnauzer": "small",
    "sealyham terrier": "small",
    "schipperke": "small",
    "italian greyhound": "small",
    "mexican hairless": "small",
    "miniature poodle": "small",
    "boston bull": "small",
    "pug": "small",
    # ---- Medium ----
    "beagle": "medium",
    "cocker spaniel": "medium",
    "english springer spaniel": "medium",
    "welsh springer spaniel": "medium",
    "brittany spaniel": "medium",
    "border collie": "medium",
    "border terrier": "medium",
    "basenji": "medium",
    "whippet": "medium",
    "australian terrier": "medium",
    "airedale": "medium",
    "bedlington terrier": "medium",
    "lakeland terrier": "medium",
    "kerry blue terrier": "medium",
    "irish terrier": "medium",
    "wire fox terrier": "medium",
    "soft-coated wheaten terrier": "medium",
    "standard schnauzer": "medium",
    "pembroke welsh corgi": "medium",
    "cardigan welsh corgi": "medium",
    "keeshond": "medium",
    "collie": "medium",
    "shetland sheepdog": "medium",
    "vizsla": "medium",
    "german short-haired pointer": "medium",
    "english setter": "medium",
    "irish setter": "medium",
    "gordon setter": "medium",
    "saluki": "medium",
    "ibizan hound": "medium",
    "chow chow": "medium",
    "basenji": "medium",
    "standard poodle": "medium",
    "groenendael": "medium",
    "malinois": "medium",
    "kelpie": "medium",
    "boxer": "medium",
    "doberman": "medium",
    "french bulldog": "small",   # brachycephalic — smaller portion
    # ---- Large ----
    "golden retriever": "large",
    "labrador retriever": "large",
    "flat-coated retriever": "large",
    "curly-coated retriever": "large",
    "chesapeake bay retriever": "large",
    "german shepherd": "large",
    "rottweiler": "large",
    "siberian husky": "large",
    "alaskan malamute": "large",
    "malamute": "large",
    "weimaraner": "large",
    "bloodhound": "large",
    "borzoi": "large",
    "afghan hound": "large",
    "rhodesian ridgeback": "large",
    "old english sheepdog": "large",
    "bouvier des flandres": "large",
    "greater swiss mountain dog": "large",
    "bernese mountain dog": "large",
    "irish wolfhound": "large",
    "scottish deerhound": "large",
    "norwegian elkhound": "large",
    "otterhound": "large",
    "walker hound": "large",
    "english foxhound": "large",
    "redbone": "large",
    "bluetick": "large",
    "black-and-tan coonhound": "large",
    "briard": "large",
    "giant schnauzer": "large",
    "samoyed": "large",
    "staffordshire bull terrier": "medium",
    "american staffordshire terrier": "large",
    "bull mastiff": "large",
    # ---- Giant ----
    "great dane": "giant",
    "saint bernard": "giant",
    "newfoundland": "giant",
    "great pyrenees": "giant",
    "tibetan mastiff": "giant",
    "leonberg": "giant",
    "komondor": "giant",
    "kuvasz": "giant",
}

# ---------------------------------------------------------------------------
# Breed-specific health notes appended to the standard plan
# ---------------------------------------------------------------------------
BREED_HEALTH_NOTES: dict[str, str] = {
    "german shepherd": "Prone to hip dysplasia — add glucosamine and omega-3 supplements.",
    "labrador retriever": "Highly food-motivated and obesity-prone; measure meals carefully.",
    "golden retriever": "Cancer risk is elevated; antioxidant-rich foods (blueberries, broccoli) are beneficial.",
    "french bulldog": "Brachycephalic — eat slowly; use a slow-feeder bowl to prevent choking.",
    "pug": "Brachycephalic and obesity-prone; limit treats strictly.",
    "dachshund": "Spinal health is critical; avoid excess weight. Feed low-fat diet.",
    "siberian husky": "Low food requirements for their size; overfeeding causes rapid weight gain.",
    "boxer": "Sensitive digestive system; avoid rapid diet changes.",
    "great dane": "Extremely bloat-prone; use a slow-feeder bowl, two meals, no exercise post-meal.",
    "doberman": "Heart disease risk (DCM) — limit grain-free diets; consult vet before diet changes.",
    "rottweiler": "Joint-support formula important from middle age.",
    "bernese mountain dog": "Short lifespan; senior food transition at 5–6 years.",
    "saint bernard": "Drool heavily after meals; ensure constant fresh water.",
    "chow chow": "Prone to thyroid issues; regular weight monitoring required.",
    "cocker spaniel": "Prone to ear infections worsened by allergies — consider limited-ingredient diet.",
    "maltese": "May refuse food; add warm water or low-sodium broth to kibble for palatability.",
    "yorkshire terrier": "Prone to hypoglycaemia; never skip meals.",
    "beagle": "Very food-motivated; measure every meal, no free-feeding.",
    "border collie": "High energy; may need 20–30% more calories on active days.",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_size_category(breed_name: str) -> str:
    """
    Return the size category ('small', 'medium', 'large', 'giant') for a breed.

    Uses an explicit breed→size map covering 120 breeds.
    Falls back to keyword matching for breeds not in the map.
    """
    if not breed_name:
        return "medium"

    normalised = breed_name.lower().strip().replace('_', ' ').replace('-', ' ')

    # 1. Exact match
    if normalised in BREED_SIZE_MAP:
        return BREED_SIZE_MAP[normalised]

    # 2. Partial match (handles cases like "labrador" matching "labrador retriever")
    for breed_key, size in BREED_SIZE_MAP.items():
        if breed_key in normalised or normalised in breed_key:
            return size

    return "medium"


def get_diet_plan(breed_name: str) -> dict:
    """
    Return the full diet plan for a breed, including any breed-specific health notes.
    """
    size = get_size_category(breed_name)
    plan = dict(diet_plans[size])  # copy so we don't mutate the template

    normalised = breed_name.lower().strip().replace('_', ' ').replace('-', ' ')
    for key, note in BREED_HEALTH_NOTES.items():
        if key in normalised or normalised in key:
            plan['breed_note'] = note
            break

    return plan