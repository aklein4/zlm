
## Bebugging
 - [ ] make sure each example is loaded once onto one device
 - [ ] investigate dtypes and nans
 - [ ] investigate hang when using flash attention

### Data
 - [x] create a single mixed and shuffled SFT dataset
 - [ ] create a single mixed and shuffled RAW dataset
  - how to handle input/output splitting and lengths?

### Models
 - [x] port ZLM from other repo
 - [x] load pretrained transformers llama checkpoint into Llama model
 - [x] load pretrained transformers llama checkpoint into ZLM model
 - [x] improve elementwise attention masking using scales and biases

### Training
 - [ ] port QOL improvements from other repo
 - [x] port ZRM training step from other repo
   - [x] remove graph breaks
   - [ ] add scans to diffusion loss calculation 
 - [x] create collators for new dataset format
 - [ ] figure out scan/checkpoint/offload for ZLM multiple transformers

### Misc.
 - [x] port improved utils from other repo 
 - [ ] create logging/printing utils