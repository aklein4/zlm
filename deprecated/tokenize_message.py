
import os

from transformers import AutoTokenizer

import utils.constants as constants

MAX_LENGTH = 128 * 3


def main():
    
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(constants.LOCAL_DATA_PATH, "tokenizer")
    )

    message = """
Solving a problem well often requires two complementary habits: thinking step by step and summarizing what you know so far. Step-by-step thinking is the engine that moves you forward; summarizing is the dashboard that tells you whether you are still on the right road. Used together, they help a person stay organized, catch mistakes early, and make steady progress even when the problem feels complicated.

Step-by-step thinking starts by turning a vague challenge into a clear target. The person states what the problem is asking, identifies the “given” information, and names the unknowns. They may rewrite the question in their own words, because misunderstanding the goal is the fastest way to waste effort. Next, they choose a plan. That plan might be a known method (like breaking a task into smaller parts, drawing a diagram, listing constraints, or testing examples), or it might be a simple decision such as “first I’ll gather facts, then I’ll compare options.” At each step, they take one small action and check what it produces. If the action does not reduce uncertainty, they adjust: try a different approach, look for missing information, or restate the problem again.

This is where summarizing becomes essential. After completing a meaningful chunk of work—reading a section, calculating a value, testing a case, or forming a hypothesis—the person pauses and summarizes what they have seen so far. A good summary is short, concrete, and testable. It might sound like: “So far, the key constraint is X,” or “The data suggests A is increasing when B decreases,” or “I’ve eliminated two options because they violate the requirements.” Summaries prevent the person from carrying around fuzzy impressions. They turn scattered observations into a small set of facts that can guide the next step.

Now we are ready to begin writing our final solution here.
""".strip()

    tokens = tokenizer(
        message,
        max_length=MAX_LENGTH,
        truncation=True,
    ).input_ids

    with open("tokens.txt", "w") as f:
        
        f.write(" ===== ORIGINAL MESSAGE ===== \n")
        f.write(message + "\n\n")

        f.write(" ===== TOKEN IDS ===== \n")
        f.write(str(tokens) + "\n\n")

        f.write("TOTAL TOKENS: " + str(len(tokens)) + "\n\n")

        f.write(" ===== DECODED MESSAGE ===== \n")
        f.write(tokenizer.decode(tokens, keep_special_tokens=True) + "\n")


if __name__ == "__main__":
    main()