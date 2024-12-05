## PoliticalBias Reposiotry
This repository contains my personal module report for DCU, 2024. The experiments are reproduced from [From Pretraining Data to Language Models to Downstream Tasks: Tracking the Trails of Political Biases Leading to Unfair NLP Models](https://arxiv.org/abs/2305.08283) presented at ACL 2023. I utilized newer models, Llama-3 and GPT-4o, for comparison, and fine-tuned GPT-2 using the POLITICS news dataset.

### Experiment of Stage 1
Code of this part can be found in [PoliLean
Public](https://github.com/BunsenFeng/PoliLean). And the requirement of enviroment as well.

### Datasets used to fine-tune
For datasets used to fineture, visit [POLITICS](https://github.com/launchnlp/politics). 

### fine_tune_code
This folder contains two Python scripts: one for generating a smaller training set and another for fine-tuning the model.

### PCT
Prompts, statements, example response and score are provided in this folder.