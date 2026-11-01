"""Code and scripts must not contain user-specific paths or names (anonymous submission)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKSLASH = chr(92)  # built from its code so this file never contains a literal backslash path
FORBIDDEN = ("C:/Users", "C:" + BACKSLASH + "Users", "/Users/", "Lohit", "Jayati")


def test_no_user_specific_strings_in_code():
    this = Path(__file__).resolve()
    offenders = []
    for folder in ("bidrig", "scripts", "tests"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.resolve() == this:
                continue
            text = path.read_text(encoding="utf-8")
            offenders += [f"{path.relative_to(ROOT)}: {s}" for s in FORBIDDEN if s in text]
    assert not offenders, offenders
