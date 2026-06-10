from __future__ import annotations

import re
from dataclasses import dataclass, field

from .normalize import NormalizedText

ATTACK = "attack"
MODERATION = "moderation"

_ATTACK_PATTERNS = [
    r"ignore (all |any |the )?(previous|prior|above|earlier) (instruction|prompt|message|rule)",
    r"(reveal|show|print|repeat|display|output) (your |the )?(system|hidden|initial) (prompt|instruction|message)",
    r"(developer|debug|god|sudo|root|unrestricted|jailbreak|dan) mode",
    r"you are (now |going to be )?(dan|an? unrestricted|an? unfiltered|jailbroken)",
    r"<\|?(im_start|im_end|system|assistant|user|endoftext)\|?>",
    r"\[/?INST\]|\[/?SYS\]|<<SYS>>|<</SYS>>",
]
_MOD_PATTERNS = [
    r"how (to|do i|can i) (make|build|create|construct) (a |an )?(bomb|explosive|weapon)",
    r"(write|create|generate) (a |me )?(virus|malware|ransomware|keylogger)",
    r"(ways|methods|painless way) to (kill myself|commit suicide|end my life)",
    r"(child|minor|underage|preteen).{0,40}(sexual|nude|porn|explicit)",
    r"(write|generate|describe) (a |me |an )?(explicit|graphic|pornographic) (sex|sexual|erotic)",
]
_SPACED_RE = re.compile(r"(?:\b\w\b[\s._\-]+){5,}\w\b")
_HOMOGLYPH_RE = re.compile(r"[\u0400-\u04ff\u0370-\u03ff]")


@dataclass
class GateResult:
    force_route: set[str] = field(default_factory=set)
    reasons: list[str] = field(default_factory=list)

    @property
    def allow_fast_skip(self) -> bool:
        return not self.force_route


def _hit(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def evaluate(norm: NormalizedText) -> GateResult:
    result = GateResult()
    text = norm.detection_text
    if _hit(_ATTACK_PATTERNS, text):
        result.force_route.add(ATTACK)
        result.reasons.append("rule:attack")
    if _hit(_MOD_PATTERNS, text):
        result.force_route.add(MODERATION)
        result.reasons.append("rule:moderation")
    if (
        norm.flags.get("excessive_zero_width")
        or norm.flags.get("suspicious_base64")
        or _SPACED_RE.search(norm.original_text)
        or _HOMOGLYPH_RE.search(norm.original_text)
    ):
        result.force_route.add(ATTACK)
        result.reasons.append("rule:obfuscation")
    return result
