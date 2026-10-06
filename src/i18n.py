import json
from pathlib import Path


ITALIAN = json.loads((Path(__file__).resolve().parents[1] / "locales" / "it.json").read_text())


def translate(text, language="en"):
    return ITALIAN.get(text, text) if language == "it" else text


def number(value, digits=0, language="en"):
    result = f"{value:,.{digits}f}"
    return result.translate(str.maketrans({",": ".", ".": ","})) if language == "it" else result
