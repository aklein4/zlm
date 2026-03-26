# Research Agenda

Research plan for March-May 2026.

TPU quota expires: ~May 31st



# iTTT

https://aklein.bearblog.dev/ittt/

**Release Name Candidates:**
- InchWorm
   - The model inches along the context, keeping constant activation sequence length 
- Dart or Arrow
   - Training only moves forward (no BPTT)


## Goals
1. Paper accepted at NeurIPS 2026
   - Paper Abstract Submission Deadline: May 4th
   - Full Paper Submission Deadline: May 6th
2. Open-source Model
   - Hobbyist-grade performance
   - Easy to setup and use


## So Far
- Pretrained ~1B models on 2B tokens at 32K sequence length
   - Showed decreasing perplexity with token position (including extrapolation) 
   - Beats dense attention baseline (1B tokens iTTT approx 2B tokens attention)
      - wins by bigger difference at later token positions


## Guidelines
- Focus on novelty:
   1. O(1) train-time memory (no BPTT)
- Be sure to mention:
   1. Native length extrapolation from inductive bias
   2. Compatibility with existing LoRA engines
   3. Ease of implementation (python-level ops only like LaCT)
- Target results:
   1. Training length performance similar to attention (even past chinchilla optimal training)
      - iTTT does particularly better than attention early in training because of inductive bias, need to show that attention doesn't overtake later 
   2. Better length extrapolation than attention
   3. Particularly good in-context learning performance
- Thorough benchmarks and baselines (good science)
   - perplexity as workhorse metric
   - real-world (RULER, LongBench) is more important than NIAH
   - Attention is gold-standard baseline, at least need to test SOTA constant state size method (LaCT? TTT-E2E?)
   - Use best version of baselines to remove doubt
- Clear-cut compute/memory metrics and graphs
- Keep methods simple and elegant
   - minimize hyper-parameters
   - minimize and ablate arbitrary design decisions 
- Include _some_ theory
   - Connections to meta-learning like MAML and Reptile
   - Muon optimizer connections to Hebbian learning
   - High-dimensional MLP -> low-noise Hebbian retrieval
- Prioritize visual diagrams for method explanations 
- Need a headline result/graph
   - Match/beat attention with constant training memory
   - Extrapolate to 1M+ token sequence length
- At least one interesting observation/finding/behavior
- Only mention philosophy (convergence, serial compute, etc) in appendix, if at all


## Experimental Setup

### Data
- Training:
   - Dolma3 Longmino at 32K sequence length
- Evaluation: 
   - Held out LongMino sequences of 32/64/128K+
   - Books 3

### Scale
Current compute+time budget gives us at least 150B-250B total tokens of 1B model training 
- Medium (~350M), XL (~1.1B) parameter
models
- Pretrain each to 25B tokens
   - That is chinchilla optimal for 1.1B
   - Much longer than chinchilla optimal for smaller models

### Details
- SmolLM2 (50K), TinyLlama (32K), or Mixtra (32K) tokenizer?
   - smaller vocab saves memory
- QK-Norm in attention?
- Default Llama init scale?
- Muon outer optimizer
- Nothing else fancy about architecture initialization

## Baselines

### Definitely
- Full attention (w/ appropriate rope theta)
- Sliding Window Attention
- Hybrid 5:1 SWA + Full attention
- LaCT
- E2E-TTT 

### If Time Permitting
1. GLA + SWA
   - Supposedly strictly worse than LaCT/E2E-TTT
2. DeltaNet + SWA
   - Supposedly strictly worse than LaCT/E2E-TTT
3. In-Place TTT
   - Less notable, more relevant for drop-in instead of training from scratch


## Evaluation

### Definitely
- Perplexity at token positions
   - up to 32K (interpolation)
   - up 128K or 1M (extrapolation)
      - use yarn-like adjustment for full attention for best performance
- Real-world benchmarks
   - RULER
   - LongBench v2
   - HELMET
- Scaling 
   - Training compute/time/memory
   - Inference compute/time/memory
- Many-shot In-Context Learning
   - ManyICLBench? 
   - LongBench v2 subset?

### If Time Permitting
- Needle in a Haystack
   - Superseded by RULER
- Basic reasoning (ARC, MMLU, etc)

## Ablations
Performed at Small (125M) or Medium (350M) scale, trained to 25B tokens? Simple evaluation with perplexity.

### Definitely
- Chunk size: 512, 1024, 2048
- LoRA rank: d_h/8, d_h/4, d_h/2
- Momentum
   - at least zero momentum since it uses less memory
- LoRA placement: Q, O, MLP up/gate?
   - equivalent state size
- Train up LoRA projection, not just down

### If time permitting
- Inner optimizers: Muon, SGD+momentum, SigNum+momentum?
- base_lr
- base state


## References

### Test-Time Training Done Right (LaCT)
https://arxiv.org/abs/2505.23884
- Trained on "Together AI Long data collections database" at 32K context length
   - 760M params for 40B tokens, 3B params for 60B tokens
- Evaluated on Book-3 perplexity and NIAH retrieval
- Baselines:
   - Full attention (1M rope theta)
   - Sliding Window Attention
   - GLA + SWA
   - DeltaNet + SWA

### End-to-End Test-Time Training for Long Context
https://arxiv.org/abs/2512.23675
- [Training recipe](./figures/e2e_ttt_recipe.png)
- Prerained on DCLM-Baseline at 8K context length
   - 125M/350M/760M/1.3B/2.7B parameters to ~chinchilla optimal token count
- Finetuned on Books at 128K context length
   - 5% number of pretraining tokens at double batch size
- Evaluated on Book-3 perplexity, NIAH retrieval
- Baselines:
   - Full attention (500K rope theta pretraining; 16K->1M, 32K->2M, 64K->5M, 128K->10M finetuning theta)
   - Sliding Window Attention
   - Hybrid 5:1 SWA + Full attention
   - Mamba 2 + SWA
   - Gated DeltaNet + SWA
   - TTT-KVB (similar to LaCT)

### Nested Learning: The Illusion of Deep Learning Architectures
https://arxiv.org/abs/2512.24695
- Relevant but convoluted methods and setup, probably not worth implementing baseline

### In-Place Test-Time Training
https://openreview.net/forum?id=dTWfCLSoyl
- Training from Scratch
   - Trained on "Together AI Long data collections database" at 32K context length
      - 500M, 1.5B, 4B (8K context length for 4B) param models
   - Baselines:
      - Full attention (only at 4B)
      - Sliding Window Attention
      - GLA
      - DeltaNet
      - LaCT + SWA
- Continued Pretraining
   - TODO

### Titans: Learning to Memorize at Test Time
https://arxiv.org/abs/2501.00663
- Foundational but LaCT is better version

### ATLAS: Learning to Optimally Memorize the Context at Test Time
https://arxiv.org/abs/2505.23735
- Foundational but LaCT is better version

### PERK: Long-Context Reasoning as Parameter-Efficient Test-Time Learning
https://arxiv.org/abs/2507.06415
- Similar to iTTT but uses truncated gradient unrolling
- Splits context into chunks and treats them as parallel batch, performs multiple gradient updates on batch
- Does not train from scratch

### qTTT
https://arxiv.org/abs/2512.13898
- Does not pretrain/finetune, only drop-in

### LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models
https://arxiv.org/abs/2502.02069
- TT adaptation, not long-context



# MonArc

https://aklein.bearblog.dev/monarc/

## Goals
1. Paper accepted at EMNLP 2026
   - ARR submission deadline (long & short papers): May 25th
2. Open-source Model
   - Research-grade performance (not as good as hobbyist)
   - Easy to setup and use

## So Far
 - Pretrained ~150M model on 50B tokens
    - Showed 2x data efficiency vs. baseline in perplexity
 - Continued pretraining ~350M model on 10B tokens
    - Showed more improvement in perplexity



# iTTT (Drop-In)

## So Far
- Demonstrated improved perplexity in plug-and-play setting against sliding-window baseline
- Continued pretraining (2B tokens) improves perplexity against plug-and-play
   - Still worse than dense attention
