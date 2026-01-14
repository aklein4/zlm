
### Data
 - [ ] create a single mixed and shuffled SFT dataset
 - [ ] create a single mixed and shuffled RAW dataset
  - how to handle input/output splitting and lengths?

### Models
 - [ ] port ZLM from other repo
 - [ ] load pretrained transformers llama checkpoint into Llama model
 - [ ] load pretrained transformers llama checkpoint into ZLM model
 - [ ] improve elementwise attention masking using scales and biases

### Training
 - [ ] port QOL improvements from other repo
 - [ ] port ZRM training step from other repo
   - [ ] remove graph breaks
   - [ ] add scans to diffusion loss calculation 
 - [ ] create collators for new dataset format
 - [ ] figure out scan/checkpoint/offload for ZLM multiple transformers

### Misc.
 - [ ] port improved utils from other repo 
 - [ ] create logging/printing utils for debugging
 - [ ] debug data loading to make sure each example is loaded once onto one device