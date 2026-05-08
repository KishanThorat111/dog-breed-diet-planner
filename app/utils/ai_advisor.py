"""
AI advisor — wraps Gemini and OpenAI to produce breed-specific nutrition advice.

Design
------
- The active provider is stored in the database (Setting.ai_provider).
- API keys are read from environment variables ONLY (never stored in the DB).
- Both providers are lazy-imported so the packages are not required at startup;
  if neither is installed, the advisor silently returns None.
- Default provider: 'gemini' (free tier: gemini-1.5-flash, 15 RPM, 1M tokens/day)
- Fallback: if the active provider's API key is missing, returns None gracefully.
"""

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (
    "You are a concise veterinary nutrition expert. "
    "When given a dog breed and its current diet plan, provide 2-3 sentences of "
    "breed-specific health and nutrition advice. Be practical. "
    "Do not repeat the diet plan — add new, useful insight about this specific breed."
)


def _build_prompt(breed: str, size: str, diet: dict) -> str:
    return (
        f"Breed: {breed} ({size} size category)\n"
        f"Current diet plan:\n"
        f"  • Food:          {diet.get('food', 'N/A')}\n"
        f"  • Meals per day: {diet.get('meals', 'N/A')}\n"
        f"  • Extras:        {diet.get('extras', 'N/A')}\n"
        f"  • Avoid:         {diet.get('avoid', 'N/A')}\n\n"
        "Give 2-3 sentences of additional breed-specific nutrition and health advice."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ai_advice(breed: str, size: str, diet: dict, provider: str) -> str | None:
    """
    Return AI-generated advice string, or None if unavailable / provider == 'none'.

    provider: 'gemini' | 'openai' | 'none'
    """
    if not provider or provider == 'none':
        return None

    try:
        if provider == 'gemini':
            return _gemini_advice(breed, size, diet)
        if provider == 'openai':
            return _openai_advice(breed, size, diet)
        logger.warning('Unknown AI provider: %s', provider)
    except Exception:
        logger.exception('AI advice failed (provider=%s, breed=%s)', provider, breed)

    return None


def provider_key_status() -> dict[str, bool]:
    """Return which providers have their API key present in the environment."""
    return {
        'gemini': bool(os.getenv('GEMINI_API_KEY')),
        'openai': bool(os.getenv('OPENAI_API_KEY')),
    }


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _gemini_advice(breed: str, size: str, diet: dict) -> str | None:
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        logger.warning('GEMINI_API_KEY not set — skipping AI advice.')
        return None

    try:
        import google.generativeai as genai  # type: ignore[import]
    except ImportError:
        logger.warning('google-generativeai not installed. Run: pip install google-generativeai')
        return None

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name='gemini-1.5-flash',
        system_instruction=_SYSTEM_PROMPT,
    )
    prompt = _build_prompt(breed, size, diet)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            max_output_tokens=200,
            temperature=0.7,
        ),
    )
    advice = response.text.strip()
    logger.info('Gemini advice retrieved for breed=%s (%d chars)', breed, len(advice))
    return advice


def _openai_advice(breed: str, size: str, diet: dict) -> str | None:
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.warning('OPENAI_API_KEY not set — skipping AI advice.')
        return None

    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError:
        logger.warning('openai not installed. Run: pip install openai')
        return None

    client = OpenAI(api_key=api_key)
    prompt = _build_prompt(breed, size, diet)
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        max_tokens=200,
        temperature=0.7,
    )
    advice = response.choices[0].message.content.strip()
    logger.info('OpenAI advice retrieved for breed=%s (%d chars)', breed, len(advice))
    return advice
