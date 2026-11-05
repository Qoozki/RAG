import re
import unicodedata

_ARABIC_CHARS = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670]")


def normalize_text(text: str) -> str:
    """
    Cleans text extracted from PDFs.
    - NFKC converts Arabic presentation forms (common in PDF extraction)
      back to normal Arabic letters.
    - Removes diacritics (tashkeel) and tatweel so words match consistently.
    """
    text = unicodedata.normalize("NFKC", text)
    text = _TASHKEEL.sub("", text)
    return text.replace("\u0640", "")

def is_arabic(text: str) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    arabic = sum(1 for c in letters if _ARABIC_CHARS.match(c))
    return arabic / len(letters) > 0.3

_CJK_CHARS = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF\u3040-\u30FF\uAC00-\uD7AF]")


def has_cjk(text: str) -> bool:
    return bool(_CJK_CHARS.search(text))


def strip_cjk(text: str) -> str:
    return _CJK_CHARS.sub("", text)
