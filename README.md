# MSLCLS
Paper: "A Multi-Stage Fine-tuning-based Method for Low-resource Cross-lingual Summarization"

## How to use
At present, the unorganized code is published first, and we will publish the organized code in the near future. There are some parameters and code segments that are not related to this work in the code we uploaded so far (it is a code segment for another study). The steps are as follows:

## Dependencies:
```
python 3.10
pytorch 1.13.1
transformers 4.26.1
pytorch-lightning 1.9.4 
```

## Training or test model:
```
python mbart_base/src/main.py --config config.yaml
```
Modify the parameters in the 'mbart_base/src/config/config.yaml‘ file to perform two-stage training
Stage 1:
```
checkpoint: None
ch2en_mode: translate  
```
Stage 2:
```
checkpoint: not None
ch2en_mode: dd_summary
is_dd: True
loss_type: 48
r: 2.0
```
More details in the comments in the config.yaml file
There are some useless configuration parameters and code segments in the code, but this will not affect the training of the model. We will delete these irrelevant code segments and configuration parameters that affect reading as soon as possible
