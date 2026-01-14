
# training data length distributions
 - The vast majority of examples have <256 input tokens and <512 output tokens. Exceptions:
     - facebook/natural_reasoning (output)
     - nvidia/AceReason-1.1-SFT (output)
     - nvidia/Nemotron-Instruction-Following
     - nvidia/Nemotron-PrismMath
     - OpenDataArena/ODA-Math-460K

# benchmark data length distributions
 - mbpp fine for 256/512
 - MMLU-Pro will be missing some for 256/512(input limit)
 - MMLU will be missing some for 256/512 (input limit)
 - Boolq fine for 256/512
 - piqa will be missing a few for 256/512 (input limit?)
 - TriviaQA will struggle to fit for 256/512 (input limit with evidence)
 - ARC will be missing some for 256/512 (input limit)
 - HellaSwag will be missing some for 256/512 (input limit)
 - WinoGrande fine for 256/512
 - SciQ fine for 256/512
 - SQuAD-v2 will be missing some for 256/512 (input limit)
 - GSM8K will be close for 256/512 (input limit)
 - Math500 will be missing a few for 256/512 (input limit)
 - Omni-Math will be missing a few for 256/512 (input limit)
 - Olympiad will be missing a few for 256/512 (input limit)

# quality
 - nvidia/OpenMathInstruct-1 has some tool calls in the output