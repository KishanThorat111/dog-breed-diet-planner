# diet_data.py

diet_plans = {
    "small": {
        "food": "High-protein small breed kibble",
        "meals": "3 small meals per day",
        "extras": "Boiled chicken, carrots"
    },
    "medium": {
        "food": "Balanced dry dog food",
        "meals": "2 meals per day",
        "extras": "Rice, eggs, vegetables"
    },
    "large": {
        "food": "Large breed joint-support formula",
        "meals": "2 large meals per day",
        "extras": "Fish oil, boiled meat"
    }
}

def get_size_category(breed_name):
    breed_name = breed_name.lower()

    small_keywords = ["pomeranian", "chihuahua", "toy", "pug"]
    large_keywords = ["retriever", "german shepherd", "husky", "labrador"]

    for word in small_keywords:
        if word in breed_name:
            return "small"

    for word in large_keywords:
        if word in breed_name:
            return "large"

    return "medium"



