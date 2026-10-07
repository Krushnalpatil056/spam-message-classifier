"""Text cleaning with regular expressions (lowercase, punctuation and stop-word removal)."""

import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

URL_PATTERN = re.compile(r"(https?://\S+|www\.\S+)")
NUMBER_PATTERN = re.compile(r"\d+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]")
SPACES_PATTERN = re.compile(r"\s+")


def clean_text(text):
    """Convert raw text into a cleaned string ready for TF-IDF."""
    text = str(text).lower()                      # 1. lowercase
    text = URL_PATTERN.sub(" urltoken ", text)    # 2. keep a marker for links (useful spam signal)
    text = NUMBER_PATTERN.sub(" numtoken ", text) # 3. keep a marker for numbers / phone numbers
    text = NON_LETTER_PATTERN.sub(" ", text)      # 4. remove punctuation and symbols
    words = [w for w in text.split() if w not in ENGLISH_STOP_WORDS]  # 5. remove stop words
    return SPACES_PATTERN.sub(" ", " ".join(words)).strip()
