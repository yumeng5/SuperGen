# SuperGen

The source code for [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding]().

## Requirements

Before running, you need to first install the required packages by typing following commands (Using a virtual environment is recommended):

```
pip3 install -r requirements.txt
```

## Overview

**SuperGen** is a **Super**vision **Gen**eration method for zero-shot learning on NLU tasks. Instead of training on task-specific data, **SuperGen** generates training data guided by label-descriptive prompts with a unidirectional language model and fine-tunes another language model on the generated data.

<img src="./SuperGen.png" width="1000px"></img>

**Training and Test Data**: Our method does not use any task-specific data (e.g., original training set). We provide our generated training set and original dev set (used as the test set) of each GLUE task under the [`data`](data) directory: `train.json` files are the generated training set (after data selection); `test.tsv` files are the original GLUE dev set (used as the test set for evaluation purpose).
**Pretraining Corpus**: We provide the processed pretraining corpus (Wikipedia and OpenWebText) for generating training data for sequence-pair tasks under the [`pretrain_corpus`](pretrain_corpus) directory; see the [README file](pretrain_corpus/README.md) there for details.

## Generating Training Data
