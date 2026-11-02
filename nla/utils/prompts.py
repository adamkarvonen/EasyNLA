"""AV prompt construction shared by the RL trainers, AV SFT, and eval/probe scripts."""


def av_chat_prompt(tokenizer, msgs):
    """The AV prompt string: chat template with a generation prompt and thinking OFF.

    Qwen3.x templates default to thinking on, which leaves an open `<think>` after the
    assistant tag; the AV is meant to answer with `<explanation>` straight away (the
    Qwen3.6 NLA card: "Apply the chat template with enable_thinking=False"). With the
    open `<think>`, the Qwen3.6 NLA (av_base + iter_000600) parsed on 25-29% of outputs,
    greedy or temperature 1, on Qwen3.6 and Qwen3.8 activations alike: the explanations
    were fine but lacked the opening `<explanation>` tag. Thinking off: 100% in all four
    cases (64 vectors each). Templates without the variable ignore it.
    """
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=False)


def build_prompt_text(prompt_msgs, inject_char, tokenizer):
    """Apply the chat template; substitute the <INJECT> placeholder with inject_char."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    return av_chat_prompt(tokenizer, msgs)
