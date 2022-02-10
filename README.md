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


## Fine-Tuning

When using the same prompt-based fine-tuning pipeline (with the same manual prompts and label words), zero-shot SuperGen even achieves better performance than few-shot LM-BFF using 32 annotated samples per class across seven GLUE classification tasks:
| Method | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | AVG |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  ------ |
| LM-BFF 32-Sample Few-Shot | 68.3/70.5 | 65.5 | 64.5 | 92.7 | 9.3 | 69.1 | 74.5 | 63.6 |
| SuperGen Zero-Shot | 72.3/73.8 | 66.1 | 73.3 | 92.8 | 32.7 | 65.3 | 82.2 | 69.4 |

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@article{meng2022generating,
  title={Generating Training Data with Language Models: Towards Zero-Shot Language Understanding},
  author={Meng, Yu and Huang, Jiaxin and Zhang, Yu and Han, Jiawei},
  journal={arXiv preprint arXiv:2010.07245},
  year={2022}
}
```
