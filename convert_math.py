"""Convert plain-text math in markdown code spans to LaTeX inline math.

Only touches backtick code spans that contain genuine math symbols.
Identifiers / commands untouched. LaTeX commands get a trailing space
so they do not swallow following letters (e.g. \\gamma R vs \\gammaR).
"""
import re
from pathlib import Path

SYM = {
    "→": r"\to ",
    "←": r"\leftarrow ",
    "↑": r"\uparrow ",
    "↓": r"\downarrow ",
    "α": r"\alpha ",
    "β": r"\beta ",
    "γ": r"\gamma ",
    "δ": r"\delta ",
    "ε": r"\epsilon ",
    "θ": r"\theta ",
    "λ": r"\lambda ",
    "μ": r"\mu ",
    "π": r"\pi ",
    "ρ": r"\rho ",
    "σ": r"\sigma ",
    "Σ": r"\sum ",
    "φ": r"\phi ",
    "∇": r"\nabla ",
    "∂": r"\partial ",
    "∫": r"\int ",
    "√": r"\sqrt ",
    "∑": r"\sum ",
    "∞": r"\infty ",
    "≤": r"\le ",
    "≥": r"\ge ",
    "≈": r"\approx ",
    "≠": r"\ne ",
    "≡": r"\equiv ",
    "·": r"\cdot ",
    "−": "-",
    "⁻": "-",
}

# Literal math chars (not used for identifiers)
MATH_TRIGGERS = list(SYM.keys())
CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")


def convert(text: str) -> str:
    def repl(m: re.Match) -> str:
        inner = m.group(1)
        if "\\" in inner:  # already LaTeX/escape sequences
            return m.group(0)
        # Must contain at least one real math symbol
        if not any(ch in inner for ch in MATH_TRIGGERS):
            return m.group(0)
        s = inner
        for k, v in SYM.items():
            s = s.replace(k, v)
        # transpose superscript T -> ^T
        s = s.replace("ᵀ", "^T ")
        s = s.replace("⁰", "^0 ").replace("¹", "^1 ").replace("²", "^2 ").replace("³", "^3 ")
        s = s.replace("⁺", "^+ ")
        # collapse multi spaces
        s = re.sub(r"[ \t]+", " ", s).strip()
        # wrap in LaTeX inline math
        return f"\\({s}\\)"

    return CODE_SPAN_RE.sub(repl, text)


for p in sorted(Path(".").glob("chapter_*/**/*.md"), key=str) + [Path("README.md")]:
    t = p.read_text(encoding="utf-8")
    nt = convert(t)
    if nt != t:
        p.write_text(nt, encoding="utf-8")
        print(f"converted: {p}")
