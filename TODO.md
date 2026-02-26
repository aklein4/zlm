
### Debugging
 - [ ] make sure that data is not duplicated when loaded on multiple devices
 - [ ] investigate dtypes and nans
   - Everything prints as float32
   - Printed precision may not match actual precision
     - https://docs.pytorch.org/xla/master/tutorials/precision_tutorial.html
     - https://docs.pytorch.org/xla/release/2.1/index.html
   - Enforcing nan-free attention does not entirely fix it


### Precision  


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