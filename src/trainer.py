########## The following part is copied from Transformers' trainer (3.4.0) ########## 

# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import collections
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.special import softmax
import torch
from packaging import version
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.distributed import DistributedSampler

import transformers
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.file_utils import WEIGHTS_NAME, is_datasets_available, is_in_notebook, is_torch_tpu_available, is_sagemaker_mp_enabled
from transformers.integrations import (
    default_hp_search_backend,
    is_comet_available,
    is_optuna_available,
    is_ray_available,
    is_tensorboard_available,
    is_wandb_available,
    run_hp_search_optuna,
    run_hp_search_ray,
)
# from transformers.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    set_seed,
)
from transformers.training_args import TrainingArguments
from transformers.utils import logging
from tqdm import tqdm, trange

_use_native_amp = False
_use_apex = False
# from apex import amp

DEFAULT_CALLBACKS = [DefaultFlowCallback]
DEFAULT_PROGRESS_CALLBACK = ProgressCallback

if is_in_notebook():
    from transformers.utils.notebook import NotebookProgressCallback

    DEFAULT_PROGRESS_CALLBACK = NotebookProgressCallback

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
# if version.parse(torch.__version__) < version.parse("1.6"):
#     from transformers.file_utils import is_apex_available

#     if is_apex_available():
#         from apex import amp
#     _use_apex = True
# else:
#     _use_native_amp = True
#     from torch.cuda.amp import autocast

if version.parse(torch.__version__) < version.parse("1.2"):
    _use_ddp_no_sync = False
else:
    _use_ddp_no_sync = True

if is_tensorboard_available():
    from transformers.integrations import TensorBoardCallback

    DEFAULT_CALLBACKS.append(TensorBoardCallback)


if is_wandb_available():
    from transformers.integrations import WandbCallback

    DEFAULT_CALLBACKS.append(WandbCallback)

if is_comet_available():
    from transformers.integrations import CometCallback

    DEFAULT_CALLBACKS.append(CometCallback)


logger = logging.get_logger(__name__)

########## The above part is copied from Transformers' trainer (3.4.0) ########## 

def default_dev_objective(metrics):
    """
    Objective used for picking the best model on development sets
    """
    if "eval_mnli/acc" in metrics:
        return metrics["eval_mnli/acc"]
    elif "eval_mnli-mm/acc" in metrics:
        return metrics["eval_mnli-mm/acc"]
    elif "eval_f1" in metrics:
        return metrics["eval_f1"]
    elif "eval_mcc" in metrics:
        return metrics["eval_mcc"]
    elif "eval_pearson" in metrics:
        return metrics["eval_pearson"]
    elif "eval_acc" in metrics:
        return metrics["eval_acc"]
 
    raise Exception("No metric founded for {}".format(metrics))


class SuperGenTrainer(transformers.Trainer):
    """
    Adding some functions based on Transformers' Trainer class.
    """
    def __init__(self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),):
        self.test_dataset = test_dataset
        super(SuperGenTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, 
            tokenizer, model_init, compute_metrics, callbacks, optimizers)

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Based on Transformers' default one, we add fixing layer option where the bottom n layers' parameters
        are fixed and only the top layers are further fine-tuned.
        """
        if self.optimizer is None:
            params = {}
            for n, p in self.model.named_parameters():
                if self.args.fix_layers > 0:
                    if 'encoder.layer' in n:
                        try:
                            layer_num = int(n[n.find('encoder.layer') + 14:].split('.')[0])
                        except:
                            print(n)
                            raise Exception("")
                        if layer_num >= self.args.fix_layers:
                            print('yes', n)
                            params[n] = p
                        else:
                            print('no ', n)
                    elif 'embeddings' in n:
                        print('no ', n)
                    else:
                        print('yes', n)
                        params[n] = p
                else:
                    if self.args.freeze_emb:
                        if 'embeddings' not in n:
                            params[n] = p
                    else:
                        params[n] = p
            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in params.items() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in params.items() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
            )
        if self.lr_scheduler is None:
            self.lr_scheduler = get_linear_schedule_with_warmup(
                self.optimizer, num_warmup_steps=int(self.args.warmup_ratio*num_training_steps), num_training_steps=num_training_steps
            )

    def update_ensemble(self, new_preds):
        for i, feature in enumerate(self.train_dataset.all_features):
            feature.update_label(list(new_preds[i]))

    # Update training data by filtering out noisy samples on which the ensembled predictions are below the threshold
    def update_train_data(self, output, threshold=0.8, momentum=0.8):
        if output is not None:
            predictions = output.predictions
            pred_probs = softmax(predictions, axis=-1)
            # Update ensemble predictions (Temporal ensemble)
            self.train_dataset.ensemble_count += 1
            self.train_dataset.ensemble_pred = (1-momentum)*pred_probs + momentum*self.train_dataset.ensemble_pred
            self.train_dataset.ensemble_pred = self.train_dataset.ensemble_pred
            pred_probs = self.train_dataset.ensemble_pred / (1-momentum**self.train_dataset.ensemble_count)
            self.update_ensemble(pred_probs)
            max_probs = np.amax(pred_probs, axis=-1)
            predictions = np.argmax(pred_probs, axis=-1)
            labels = np.array([feature.label for feature in self.train_dataset.features])
            select_pos = (labels == predictions) & (max_probs > threshold)
            for i in range(self.train_dataset.num_labels):
                num_train = np.sum((labels == i) & select_pos)
                # If there are too few valid training data for some class, use half of the entire training data sorted by confidence
                if num_train < 16:
                    pred_probs_class = pred_probs[:, i]
                    sorted_idx = np.argsort(-pred_probs_class[labels == i])
                    threshold_idx = sorted_idx[int(0.5*sum(labels == i))]
                    threshold = pred_probs_class[labels == i][threshold_idx]
                    select_pos = select_pos | ((labels == i) & (pred_probs_class > threshold))
                    num_train = np.sum((labels == i) & select_pos)
            valid_idx = self.train_dataset.remain_idx.intersection(set(np.where(select_pos)[0]))
        else:
            valid_idx = self.train_dataset.remain_idx
        target_size = min(self.total_train_batch_size * self.args.eval_steps, len(self.train_dataset.all_features))
        # If remaining valid samples are not enough to form the training data for the next update interval, 
        # use all remaining ones and restart from the beginning to pick training samples 
        if target_size > len(valid_idx):
            select_idx = np.where(select_pos)[0][:(target_size-len(valid_idx))]
            self.train_dataset.features = [self.train_dataset.features[i] for i in valid_idx] + [self.train_dataset.features[i] for i in select_idx]
            self.train_dataset.remain_idx = set([i for i in range(len(self.train_dataset.all_features))])
            self.train_dataset.remain_idx = self.train_dataset.remain_idx.difference(set(select_idx))
        # Pick valid training samples and update remaining indices
        else:
            select_idx = list(valid_idx)[:target_size]
            self.train_dataset.features = [self.train_dataset.features[i] for i in select_idx]
            self.train_dataset.remain_idx = self.train_dataset.remain_idx.difference(set(select_idx))
        self.train_dataset.size = len(self.train_dataset.features)
    
    def reset_train_data(self):
        self.train_dataset.features = self.train_dataset.all_features.copy()
        self.train_dataset.size = len(self.train_dataset.features)

    def train(self, model_path=None, dev_objective=None):
        """
        Main training entry point.

        The training logic is directly borrowed from transformers.Trainer (version 3.0.2).
        Add early stopping.
        """
        self.best_dir = None
        self.objective = -float("inf")
        self.dev_objective = dev_objective if dev_objective is not None else default_dev_objective

        # Data loading.
        train_dataloader = self.get_train_dataloader()
        num_update_steps_per_epoch = len(train_dataloader) // self.args.gradient_accumulation_steps 
        if num_update_steps_per_epoch == 0:
            num_update_steps_per_epoch = 1
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
        else:
            t_total = int(len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs)
            self.args.max_steps = t_total

        self.create_optimizer_and_scheduler(num_training_steps=t_total)
        optimizer = self.optimizer
        scheduler = self.lr_scheduler

        # Check if saved optimizer or scheduler states exist
        if (
            model_path is not None
            and os.path.isfile(os.path.join(model_path, "optimizer.pt"))
            and os.path.isfile(os.path.join(model_path, "scheduler.pt"))
        ):
            # Load in optimizer and scheduler states
            optimizer.load_state_dict(
                torch.load(os.path.join(model_path, "optimizer.pt"), map_location=self.args.device)
            )
            scheduler.load_state_dict(torch.load(os.path.join(model_path, "scheduler.pt")))

        model = self.model

        if self.args.fp16 and _use_apex:
            if not transformers.is_apex_available():
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            model, optimizer = amp.initialize(model, optimizer, opt_level=self.args.fp16_opt_level)
        
        # Multi-gpu training (should be after apex fp16 initialization)
        if self.args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Distributed training (should be after apex fp16 initialization)
        if self.args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=True,
            )

        # Train
        if transformers.is_torch_tpu_available():
            self.total_train_batch_size = self.args.train_batch_size * xm.xrt_world_size()
        else:
            self.total_train_batch_size = (
                self.args.train_batch_size
                * self.args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if self.args.local_rank != -1 else 1)
            )
        logger.info("***** Running training *****")
        logger.info("  Num total examples = %d", self.num_examples(train_dataloader))
        logger.info("  Instantaneous batch size per device = %d", self.args.per_device_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", self.total_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        self.global_step = 0
        self.epoch = 0
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        # Check if continuing training from a checkpoint
        if model_path is not None:
            # set global_step to global_step of last saved checkpoint from model path
            try:
                self.global_step = int(model_path.split("-")[-1].split("/")[0])
                epochs_trained = self.global_step // (len(train_dataloader) // self.args.gradient_accumulation_steps)
                steps_trained_in_current_epoch = self.global_step % (
                    len(train_dataloader) // self.args.gradient_accumulation_steps
                )

                logger.info("  Continuing training from checkpoint, will skip to saved global_step")
                logger.info("  Continuing training from epoch %d", epochs_trained)
                logger.info("  Continuing training from global step %d", self.global_step)
                logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
            except ValueError:
                self.global_step = 0
                logger.info("  Starting fine-tuning.")

        tr_loss = torch.tensor(0.0).to(self.args.device)
        logging_loss_scalar = 0.0
        model.zero_grad()
        epoch = 0
        while True:
            print(f"** Epoch: {epoch} **")
            self.reset_train_data()
            if epoch > 0:
                print(f"Evaluating on training set for temporal ensembling")
                output = self.evaluate(eval_dataset=self.train_dataset, log=False)
            else:
                output = None
            self.update_train_data(output, threshold=self.args.threshold, momentum=self.args.momentum)
            train_dataloader = self.get_train_dataloader()
            
            logger.info("Num examples used = %d", self.num_examples(train_dataloader))

            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)

            if transformers.is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [self.args.device]).per_device_loader(
                    self.args.device
                )
                epoch_iterator = tqdm(parallel_loader, desc="Train Iteration")
            else:
                epoch_iterator = tqdm(train_dataloader, desc="Train Iteration")

            # Reset the past mems state at the beginning of each epoch if necessary.
            if self.args.past_index >= 0:
                self._past = None
            
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    continue
                inputs["rampup_ratio"] = self.train_dataset.ensemble_count / self.args.temp_ensemble_rampup
                loss = self.training_step(model, inputs)
                tr_loss += loss

                if (step + 1) % self.args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= self.args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
                ):
                    if self.args.fp16 and _use_native_amp:
                        self.scaler.unscale_(optimizer)
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                    elif self.args.fp16:
                        norm = torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.max_grad_norm)
                    else:
                        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)

                    if transformers.is_torch_tpu_available():
                        xm.optimizer_step(optimizer)
                    elif self.args.fp16 and _use_native_amp:
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        optimizer.step()

                    scheduler.step()
                    model.zero_grad()
                    self.global_step += 1
                    self.epoch = epoch + (step + 1) / len(epoch_iterator)

                    if (self.args.logging_steps > 0 and self.global_step % self.args.logging_steps == 0) or (
                        self.global_step == 1 and self.args.logging_first_step
                    ):
                        logs = {}
                        tr_loss_scalar = tr_loss.item()
                        logs["loss"] = (tr_loss_scalar - logging_loss_scalar) / self.args.logging_steps
                        logs["norm"] = norm.item()
                        # backward compatibility for pytorch schedulers
                        logs["learning_rate"] = (
                            scheduler.get_last_lr()[0]
                            if version.parse(torch.__version__) >= version.parse("1.4")
                            else scheduler.get_lr()[0]
                        )
                        logging_loss_scalar = tr_loss_scalar
                        self.state.global_step = self.global_step
                        self.log(logs)
                
                if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                    epoch_iterator.close()
                    break

                metrics = None
                if self.args.evaluate_during_training and (self.global_step % self.args.eval_steps == 0):
                    output = self.evaluate(eval_dataset=self.test_dataset, desc="Evalution @ Test")
                    metrics = output.metrics
                    objective = self.dev_objective(metrics)
                    print(f"Test objective: {objective}")
                    break

            epoch += 1
            if self.args.max_steps > 0 and self.global_step > self.args.max_steps:
                # train_iterator.close()
                break
            if self.args.tpu_metrics_debug or self.args.debug:
                # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                xm.master_print(met.metrics_report())

        if self.args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        return TrainOutput(self.global_step, tr_loss / self.global_step, logs), self.objective


    """
    Difference compared to original implementation: return output instead of output.metrics (so there is also the logits)
    """
    def evaluate(self, eval_dataset: Optional[Dataset] = None, log=True, desc="Evaluation") -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are
        task-dependent (pass it to the init :obj:`compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (:obj:`Dataset`, `optional`):
                Pass a dataset if you wish to override :obj:`self.eval_dataset`. If it is an :obj:`datasets.Dataset`,
                columns not accepted by the ``model.forward()`` method are automatically removed. It must implement
                the :obj:`__len__` method.

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions.
        """
        if eval_dataset is not None and not isinstance(eval_dataset, collections.abc.Sized):
            raise ValueError("eval_dataset must implement __len__")

        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        output = self.prediction_loop(eval_dataloader, description=desc)

        if log:
            self.log(output.metrics)

        if self.args.tpu_metrics_debug or self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        return output
