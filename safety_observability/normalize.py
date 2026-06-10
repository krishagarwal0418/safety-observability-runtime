from __future__ import annotations

import base64
import binascii
import hashlib
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from html import unescape

_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff]")
_TAG_CHARS_RE = re.compile(r"[\U000E0000-\U000E007F]")
_WHITESPACE_RE = re.compile(r"\s+")
_BASE64_SPAN_RE = re.compile(r"(?<![A-Za-z0-9+/=])[A-Za-z0-9+/]{24,}={0,2}(?![A-Za-z0-9+/=])")
_REPEAT_SEP_RE = re.compile(r"([\-=_*~#.])\1{9,}")


@dataclass
class NormalizedText:
    original_text: str
    model_text: str
    detection_text: str
    text_hash: str
    flags: dict[str, object] = field(default_factory=dict)


def _decode_base64_spans(spans: list[str]) -> list[str]:
    decoded: list[str] = []
    for span in spans[:4]:
        if len(span) % 4 != 0:
            continue
        try:
            raw = base64.b64decode(span, validate=True)
            text = raw.decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            continue
        printable = sum(1 for ch in text if ch.isprintable() or ch in " \n\t")
        if text and printable / len(text) >= 0.8:
            decoded.append(text[:512])
    return decoded


def normalize(text: str) -> NormalizedText:
    if text is None or not str(text).strip():
        raise ValueError("text is empty")
    original = str(text)
    flags: dict[str, object] = {}
    work = unicodedata.normalize("NFKC", original)
    zw = len(_ZERO_WIDTH_RE.findall(work))
    tags = len(_TAG_CHARS_RE.findall(work))
    work = _ZERO_WIDTH_RE.sub("", work)
    work = _TAG_CHARS_RE.sub("", work)
    flags["zero_width_removed"] = zw
    flags["tag_chars_removed"] = tags
    flags["excessive_zero_width"] = zw >= 5
    work = urllib.parse.unquote(unescape(work))
    flags["excessive_separators"] = bool(_REPEAT_SEP_RE.search(work))
    work = _WHITESPACE_RE.sub(" ", work).strip()
    model_text = work
    detection_text = work.lower()
    spans = _BASE64_SPAN_RE.findall(work)
    flags["base64_span_count"] = len(spans)
    flags["suspicious_base64"] = bool(spans)
    decoded = _decode_base64_spans(spans)
    if decoded:
        flags["base64_decoded_count"] = len(decoded)
        detection_text += " " + " ".join(d.lower() for d in decoded)
    return NormalizedText(
        original_text=original,
        model_text=model_text,
        detection_text=detection_text,
        text_hash=hashlib.sha256(model_text.encode("utf-8")).hexdigest(),
        flags=flags,
    )
