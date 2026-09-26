import os
from dotenv import load_dotenv

load_dotenv()

def get_gemini_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set."
        )

    return api_key


def get_gemini_model() -> str:
    return os.getenv(
        "GEMINI_MODEL",
        "gemini-3.5-flash-lite",
    )
