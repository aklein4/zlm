# Research Agenda

Research plan for March-May 2026.

TPU quota expires: ~May 31st

# iTTT

## Goals
1. Paper accepted at NeurIPS 2026
   - Paper Abstract Submission Deadline: May 4th
   - Full Paper Submission Deadline: May 6th
2. Open-source Model
   - Hobbyist-grade performance
   - Easy to setup and use

## So Far

### Existing Models
 - Demonstrated improved perplexity in plug-and-play setting against sliding-window baseline
 - Continued pretraining (2B tokens) improves perplexity against plug-and-play
    - Still worse than dense attention

### Training From Scratch
 - Pretrained ~1B models on 2B tokens at 32K sequence length
    - Showed decreasing perplexity with token position (including extrapolation) 
    - Beats dense attention baseline


# MonArc

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
