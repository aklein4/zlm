
from transformers import AutoTokenizer

URL = 'HuggingFaceTB/SmolLM2-360M'

def main():

    tokenizer = AutoTokenizer.from_pretrained(URL)
    if tokenizer.pad_token is None:
        print("Adding pad token...")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    
    tokenizer.save_pretrained('./tokenizer/')


if __name__ == '__main__':
    main()