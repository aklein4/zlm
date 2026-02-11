
import string


def format_chat(messages):
    assert isinstance(messages, (list, tuple)), "messages must be a list or tuple"
    assert len(messages) >= 1, "messages must contain at least one message"

    # recursive call for batch
    if isinstance(messages[0], (list, tuple)):
        return [format_chat(msgs) for msgs in messages]

    # extract roles and contents
    roles = [msg["role"] for msg in messages]
    content = [msg["content"].strip() for msg in messages]

    # define formatting lambdas
    user_only = lambda c: f"<|im_start|>user\n{c}<|im_end|>\n<|im_start|>assistant\n"
    system_and_user = lambda c1, c2: f"<|im_start|>system\n{c1}<|im_end|>\n<|im_start|>user\n{c2}<|im_end|>\n<|im_start|>assistant\n"
    assistant_only = lambda c: c+"<|im_end|>"

    # input-only roles
    if roles == ["user"]:
        return user_only(content[0])

    if roles == ["system", "user"]:
        if content[0] == "":
            return user_only(content[1])
        return system_and_user(content[0], content[1])

    # input-output roles
    if roles == ["user", "assistant"]:
        return user_only(content[0]), assistant_only(content[1])

    if roles == ["system", "user", "assistant"]:
        if content[0] == "":
            return user_only(content[1]), assistant_only(content[2])
        return system_and_user(content[0], content[1]), assistant_only(content[2])
    
    raise ValueError(f"Unsupported message roles: {roles}")


def format_no_cot(x, y, answer):
    return format_chat(
        [
            {"role": "system", "content": "Place the final answer to the following question inside of a \\boxed{} command. This must appear at the start of your response before any other text."},
            {"role": "user", "content": x},
            {"role": "assistant", "content": "\\boxed{"+str(answer)+"}"+f"\n{y}"},
        ]
    )


def format_cot(x, y, answer):
    return format_chat(
        [
            {"role": "system", "content": "Place the final answer to the following question inside of a \\boxed{} command. This must come at the end of your response, and no other text should come after it."},
            {"role": "user", "content": x},
            {"role": "assistant", "content": f"{y}\nFinal answer: \\boxed"+"{"+str(answer)+"}"},
        ]
    )


def mcqa_question(prompt, choices):
    x = prompt + "\n"

    for i, choice in enumerate(choices):
        letter = string.ascii_uppercase[i]
        x += f"{letter}: {choice}\n"
    
    return x


def remove_pad(text):
    if isinstance(text, (list, tuple)):
        return [remove_pad(t) for t in text]
    
    if "<|im_end|>" not in text:
        return text

    return text.split("<|im_end|>")[0] + "<|im_end|>"