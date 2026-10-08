import json
from pathlib import Path

import pytest

hr = pytest.importorskip("gptoss_ref.harmony_render")
CORPUS = Path(__file__).resolve().parents[1] / "corpus/golden_v1.jsonl"


def _item(name):
    return next(json.loads(l) for l in CORPUS.read_text().splitlines() if json.loads(l)["id"] == name)


def _generated_text(tokens):
    """The generated tokens, decoded, with prompt stretches shown as '|'."""
    enc = hr.encoding()
    mask = hr.generated_mask(tokens)
    out, run = [], []
    for p, g in enumerate(mask):
        if g:
            run.append(tokens[p + 1])
        elif run:
            out.append(enc.decode(run))
            run = []
    if run:
        out.append(enc.decode(run))
    return out


def test_tool_call_turn_marks_exactly_what_the_model_writes():
    segments = _generated_text(hr.render(_item("tool-weather")))
    # 1st segment: the turn's analysis, then the model itself moves on to the tool call and stops at <|call|>.
    assert segments[0].startswith("<|channel|>analysis<|message|>Need to call get_weather")
    assert "<|end|><|start|>assistant to=functions.get_weather<|channel|>commentary" in segments[0]
    assert segments[0].endswith('{"location":"Tokyo","unit":"celsius"}<|call|>')
    # The tool result and the `<|start|>assistant` the server appends after it are prompt; then analysis + final.
    assert segments[1].startswith("<|channel|>analysis<|message|>The tool returned")
    assert "<|start|>assistant<|channel|>final<|message|>It's currently 18" in segments[1]
    assert segments[1].endswith("<|return|>")
    assert len(segments) == 2
    assert not any("light rain\", \"humidity" in s for s in segments)     # tool output is never "generated"


def test_render_keeps_analysis_and_ends_with_return():
    enc = hr.encoding()
    for line in CORPUS.read_text().splitlines():
        item = json.loads(line)
        text = enc.decode(hr.render(item))
        n_analysis = sum(1 for m in item["messages"] if m.get("channel") == "analysis")
        assert text.count("<|channel|>analysis") == n_analysis, item["id"]
        assert text.endswith("<|return|>"), item["id"]
        assert f"Current date: {item['conversation_start_date']}" in text, item["id"]
