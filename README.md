# SuperGen

The source code for [Generating Training Data with Language Models: Towards Zero-Shot Language Understanding](https://arxiv.org/abs/2202.04538), published in NeurIPS 2022.

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

The generated training set used in the paper are provided as `train.json` files under each task directory; you should be able to obtain very similar generated data by following the steps below:

**Data Generation**: The entry script for generating training data for GLUE tasks is [`gen_train_data.py`](gen_train_data.py). The basic usage is
```
python gen_train_data.py --task $TASK --label $LABEL --save_dir $SAVE_DIR --num_gen $NUM_GEN
```
You can generate training data of each label either by setting individual label name `$LABEL` one at a time or by setting `$LABEL=all` to generate data for all labels (this will still be done sequentially). You may want to set `$NUM_GEN` to be larger than the desired training set size, as only those texts with the highest generated probability will be used to form the final training set.

**Data Selection**: After generating the training data, the final training set can be constructed by running the following:
```
python src/gen_utils.py --task $TASK --num_select_samples $NUM_SELECT \
                        --read_dir $SAVE_DIR --save_dir $DATA_DIR
```

**Example**: We provide an example script [`run_gen.sh`](run_gen.sh) that includes the entire generation process for all GLUE tasks under the setting described in the paper.

## Fine-Tuning

The entry script for fine-tuning on generated data is [`finetune.py`](finetune.py). The basic usage is
```
python finetune.py \
    --task_name $TASK \
    --data_dir data/$TASK \
    --overwrite_output_dir \
    --do_train \
    --do_predict \
    --smooth $SM \
    --momentum $MOMENT \
    --eval_steps $INTERVAL \
    --threshold $TH \
    --reg_weight $REG \
    --temp_ensemble_rampup $RAMP \
    --model_name_or_path $MODEL \
    --max_seq_length 128 \
    --first_sent_limit 100 \
    --per_device_train_batch_size $BS \
    --learning_rate $LR \
    --num_train_epochs 3 \
    --output_dir $OUT_DIR \
    --template $TEMPLATE \
    --mapping $MAPPING \
    --warmup_ratio 0.1 \
    --save_at_last \
```

**Example**: We provide an example script [`run_finetune.sh`](run_finetune.sh) with command line arguments set up for all GLUE tasks under the setting described in the paper.

**Results**: When using the same prompt-based fine-tuning pipeline (with the same manual prompts and label words), zero-shot SuperGen even achieves better performance than few-shot [LM-BFF](https://github.com/princeton-nlp/LM-BFF) using 32 annotated samples per class across seven GLUE classification tasks:
| Method | MNLI-m/mm | QQP | QNLI | SST-2 | CoLA | RTE | MRPC | AVG |
| ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |  ------ |
| LM-BFF 32-Sample Few-Shot | 68.3/70.5 | 65.5 | 64.5 | 92.7 | 9.3 | 69.1 | 74.5 | 63.6 |
| SuperGen Zero-Shot | 72.3/73.8 | 66.1 | 73.3 | 92.8 | 32.7 | 65.3 | 82.2 | 69.4 |

## Acknowledgement

Some scripts in this repository are adapted from [COCO-LM](https://github.com/microsoft/COCO-LM) (for COCO-LM model), [LM-BFF](https://github.com/princeton-nlp/LM-BFF) (for prompt-based fine-tuning) and [huggingface transformers](https://github.com/huggingface/transformers) (for text generation and GLUE processor/trainer).

## Citations

Please cite the following paper if you find the code helpful for your research.
```
@inproceedings{meng2022generating,
  title={Generating Training Data with Language Models: Towards Zero-Shot Language Understanding},
  author={Meng, Yu and Huang, Jiaxin and Zhang, Yu and Han, Jiawei},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
