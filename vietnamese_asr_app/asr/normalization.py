"""Vietnamese Text Normalization Module.

Provides deterministic NFC Unicode normalization, lowercasing,
punctuation stripping, and whitespace collapse for ASR evaluation and preprocessing.
"""

import re
import unicodedata

# Unicode punctuation and common non-alphanumeric symbols in Vietnamese text
PUNCTUATION_REGEX = re.compile(r"""[.,!?;:"'“”‘’()\[\]{}–—\-_/\\|`~@#$%^&*+=<>]+""", re.UNICODE)
WHITESPACE_REGEX = re.compile(r"\s+", re.UNICODE)


def normalize_vietnamese_text(text: str, strip_punctuation: bool = True, lowercase: bool = True) -> str:
    """Normalize Vietnamese transcript text deterministically.

    Args:
        text: Raw input transcript.
        strip_punctuation: Whether to remove punctuation marks.
        lowercase: Whether to convert text to lower case.

    Returns:
        Canonical normalized string.
    """
    if text is None:
        return ""
    
    # 1. Unicode NFC normalization (crucial for Vietnamese combining marks)
    norm_text = unicodedata.normalize("NFC", str(text))
    
    # 2. Lowercase conversion
    if lowercase:
        norm_text = norm_text.lower()
    
    # 3. Strip punctuation
    if strip_punctuation:
        norm_text = PUNCTUATION_REGEX.sub(" ", norm_text)
        
    # 4. Collapse multiple whitespace and strip edges
    norm_text = WHITESPACE_REGEX.sub(" ", norm_text).strip()
    return norm_text
