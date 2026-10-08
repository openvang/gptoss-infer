"""Check that the checkpoint's tokenizer.json tokenizes exactly like harmony's encoding and tiktoken o200k_harmony.

    python reference/scripts/tokenizer_check.py --model-dir /path/to/gpt-oss-20b

A C++ server will load tokenizer.json (e.g. through tokenizers-cpp), while the model was trained with tiktoken's
o200k_harmony. This script compares the three on the corpus texts plus edge cases (digits, whitespace runs, CJK,
emoji, code, invalid-looking unicode) and fails on any difference in ids or in the decode round trip.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gptoss_ref import harmony_render as hr  # noqa: E402

EDGE_CASES = [
    "", " ", "  ", "\n", "\n\n\n", "\t\t x", "a  b   c    d", "trailing space ", " leading space",
    "1234567890", "3.14159265358979", "1,000,000 and 1.000.000", "x=1+2*3/4-5", "(12)(345)(6789)",
    "naïve café résumé Ångström", "Straße STRASSE ß", "İstanbul ıi", "ﬁ ligature", "e\u0301 vs é",
    "日本語のテキストと漢字、ひらがな、カタカナ。", "中文文本：机器学习，深度学习。", "한국어 텍스트입니다.",
    "العربية من اليمين إلى اليسار", "עברית", "हिन्दी पाठ", "ไทย", "Русский текст", "Ελληνικά",
    "emoji 😀🎉👩‍💻🇫🇷", "zero\u200bwidth\u200djoin", "\u00a0nbsp\u2009thin\u202fnarrow",
    "def f(x):\n    return x ** 2  # square\n", "```python\nprint('hi')\n```", "<div class=\"a\">&amp;</div>",
    "https://example.com/path?q=1&r=two#frag", "user@example.com", "C:\\Users\\name\\file.txt",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", "!!!???...,,,;;;",
    "<|start|> is plain text here", "The year 2026, version 0.6.30, and 4096-token windows.",
]


def corpus_texts(path):
    for line in Path(path).read_text().splitlines():
        if line.strip():
            for m in json.loads(line)["messages"]:
                yield m["content"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--corpus", default=str(Path(__file__).resolve().parents[1] / "corpus/golden_v1.jsonl"))
    args = ap.parse_args()

    import tiktoken
    from tokenizers import Tokenizer

    hf = Tokenizer.from_file(str(Path(args.model_dir) / "tokenizer.json"))
    enc = hr.encoding()
    tk = tiktoken.get_encoding("o200k_harmony")

    # User-supplied text must never become control tokens: "<|start|>" typed by a user is five-plus ordinary
    # tokens, not id 200006. Report what tokenizer.json does by default, then compare plain-text encodings.
    probe = "<|start|>assistant<|channel|>final<|message|>"
    default_ids = hf.encode(probe, add_special_tokens=False).ids
    specials = sorted(i for i in default_ids if i >= 199998)
    print(f"tokenizer.json default on {probe!r}: {default_ids} -> "
          f"{'MAPS USER TEXT TO SPECIAL IDS ' + str(specials) if specials else 'plain text'}")
    hf.encode_special_tokens = True        # tokenizers >= 0.19: added/special tokens in raw text stay plain text

    texts = EDGE_CASES + list(corpus_texts(args.corpus))
    failures = 0
    for text in texts:
        a = hf.encode(text, add_special_tokens=False).ids
        b = enc.encode(text, disallowed_special=())
        c = tk.encode(text, disallowed_special=())
        problems = []
        if not (a == b == c):
            problems.append(f"ids differ: tokenizer.json {a[:12]}… harmony {b[:12]}… tiktoken {c[:12]}…")
        if hf.decode(a, skip_special_tokens=False) != text:
            problems.append("tokenizer.json decode does not round-trip")
        if problems:
            failures += 1
            print(f"FAIL {text[:60]!r}: " + "; ".join(problems))
    print(f"{len(texts) - failures}/{len(texts)} texts identical across tokenizer.json, harmony and tiktoken")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
