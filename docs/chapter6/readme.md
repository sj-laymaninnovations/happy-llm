# Chapter 6 LLM Training Based on Transformers

Note: The core content of this chapter is to implement LLM pre-training and fine-tuning based on the transformers framework

1. Framework brief description:
   1. transformers
   2. deepspeed
   3. peft
   4. wandb
   5. tokenizers
2. LLM pre-training based on transformers
   1. Word participle training
   2. Dataset construction
   3. Model building/inheriting pre-trained models
   4. Construct Trainer for training
3. LLM SFT/downstream task fine-tuning based on transformers
   1. Word participle training
   2. Dataset construction
   3. LoRA configuration
   4. Inherit the pre-trained model
   5. Construct Trainer for training