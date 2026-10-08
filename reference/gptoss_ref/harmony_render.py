"""Render golden-corpus items to token ids with OpenAI's harmony library (the format's source of truth).

A corpus item is one JSON object:
    {"id": str, "reasoning_effort": "low"|"medium"|"high", "conversation_start_date": "YYYY-MM-DD",
     "developer": {"instructions": str, "tools": [{"name", "description", "parameters"}]},      # optional
     "messages": [{"role": "user"|"assistant"|"tool", "content": str,
                   "channel": "analysis"|"commentary"|"final", "recipient": str,              # optional
                   "content_type": "<|constrain|>json", "name": "functions.x"}]}              # optional
The date is pinned per item: the HF chat template would insert today's date and break reproducibility.
"""
from openai_harmony import (Author, Conversation, DeveloperContent, HarmonyEncodingName, Message, ReasoningEffort,
                            RenderConversationConfig, Role, SystemContent, ToolDescription, load_harmony_encoding)

EFFORT = {"low": ReasoningEffort.LOW, "medium": ReasoningEffort.MEDIUM, "high": ReasoningEffort.HIGH}
_ENC = None


def encoding():
    global _ENC
    if _ENC is None:
        _ENC = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    return _ENC


def conversation(item):
    system = (SystemContent.new()
              .with_reasoning_effort(EFFORT[item.get("reasoning_effort", "medium")])
              .with_conversation_start_date(item["conversation_start_date"]))
    msgs = [Message.from_role_and_content(Role.SYSTEM, system)]
    dev = item.get("developer")
    if dev:
        content = DeveloperContent.new()
        if dev.get("instructions"):
            content = content.with_instructions(dev["instructions"])
        if dev.get("tools"):
            content = content.with_function_tools([
                ToolDescription.new(t["name"], t["description"], parameters=t.get("parameters"))
                for t in dev["tools"]])
        msgs.append(Message.from_role_and_content(Role.DEVELOPER, content))
    for m in item["messages"]:
        if m["role"] == "user":
            msg = Message.from_role_and_content(Role.USER, m["content"])
        elif m["role"] == "assistant":
            msg = Message.from_role_and_content(Role.ASSISTANT, m["content"])
        elif m["role"] == "tool":
            msg = Message.from_author_and_content(Author.new(Role.TOOL, m["name"]), m["content"])
        else:
            raise ValueError(f"{item['id']}: unknown role {m['role']!r}")
        if m.get("channel"):
            msg = msg.with_channel(m["channel"])
        if m.get("recipient"):
            msg = msg.with_recipient(m["recipient"])
        if m.get("content_type"):
            msg = msg.with_content_type(m["content_type"])
        msgs.append(msg)
    return Conversation.from_messages(msgs)


START, CHANNEL, MESSAGE = 200006, 200005, 200008
ENDS = {200007, 200002, 200012}            # <|end|>, <|return|>, <|call|>


def generated_mask(tokens):
    """Bool [n-1]: True where the token at p+1 is one the model itself generates at inference.

    That is everything in an assistant message after its role token, through its end token, plus the
    `<|start|>assistant` of an assistant message that directly follows another one (the model writes those
    when it moves from analysis to a tool call or to final). The `<|start|>assistant` that opens a turn is
    prompt: the server appends it. gpt-oss was trained with loss on generated tokens only, so its predictions
    at other positions (inside system, developer, user and tool messages) are untrained and can be poor.
    That does not affect the KL gate, which checks numerical fidelity everywhere.
    """
    toks = [int(t) for t in tokens]
    gen = [False] * len(toks)
    enc = encoding()
    prev_assistant = False
    i = 0
    while i < len(toks):
        if toks[i] != START:
            i += 1
            continue
        j = i + 1
        while j < len(toks) and toks[j] not in (CHANNEL, MESSAGE):
            j += 1
        is_assistant = enc.decode(toks[i + 1:j]).strip().startswith("assistant")
        end = j
        while end < len(toks) and toks[end] not in ENDS:
            end += 1
        if is_assistant:
            first = i if prev_assistant else i + 2          # i + 1 is the role token "assistant"
            for t in range(first, min(end, len(toks) - 1) + 1):
                gen[t] = True
        prev_assistant = is_assistant
        i = end + 1
    return [gen[p + 1] for p in range(len(toks) - 1)]


def render(item):
    """Token ids for the whole conversation. The last assistant message ends in <|return|> (final) or <|call|>.

    auto_drop_analysis is OFF: the library's default drops every analysis message that precedes a final one,
    including the reasoning of the turn being generated. At inference the model writes analysis, tool calls and
    the final answer into one growing context, so the golden must contain that same sequence. Dropping CoT from
    *earlier* turns is the caller's job: write those turns without analysis messages in the corpus.
    """
    cfg = RenderConversationConfig(auto_drop_analysis=False)
    return list(encoding().render_conversation_for_training(conversation(item), cfg))
