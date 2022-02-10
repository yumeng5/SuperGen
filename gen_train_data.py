# Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/text-generation/run_generation.py

import argparse
import logging
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import string
import json
import os
from nltk.tokenize import sent_tokenize

from transformers import (
    CTRLTokenizer,
)
from src.gen_with_reward import CTRLLMHeadModelWithRepReward
from src.gen_utils import sort_score, save

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Single-sequence or sequence-pair task
task_type_mapping = {
    "mnli": "pair",
    "qqp": "pair",
    "qnli": "pair",
    "sst-2": "single",
    "cola": "single",
    "rte": "pair",
    "mrpc": "pair",
}

# Control code used by CTRL as the starting token
control_code_mapping = {
    "mnli": "Wikipedia",
    "qqp": "Links",
    "qnli": "Links",
    "sst-2": "Reviews",
    "cola": "Links",
    "rte": "Wikipedia",
    "mrpc": "Wikipedia",
}

# If specified, generation will start with one of the given options
fix_start_mapping = {
    "sst-2": ["The movie", "The film", "This movie", "This film",
              "the movie", "the film", "this movie", "this film"],
    "cola": ['Such', 'Again', 'Until', 'Her', 'Any', 'These', 'Where', 'She', 'The', 'We',
             'Both', 'Under', 'At', 'Of', 'Doing', "You're", 'More', 'Between', 'All',
             'While', 'As', 'Our', 'Just', 'Once', 'His', 'Other', 'Most', 'In', 'My', 'Ours',
             'Before', 'When', 'He', 'There', 'Here', 'So', 'Because', 'You', 'Over',
             'During', 'Above', 'They', 'To', 'For', 'But', 'Only', 'Those', 'Against',
             'Your', 'After', 'Now', 'An', 'Too', 'Same', 'Its', 'From', 'Being', 'With',
             'A', 'Their', 'Each', "She's", 'It', 'No', 'Then', "It's", "You've", 'Some', 
             'Few', 'This', 'If', 'By', 'I'],
}

# Valid stop tokens used to terminate a sequence
stop_tokens_mapping = {
    "mnli": ['. '],
    "qqp": ['? ', '?\n'],
    "qnli": ['. '],
    "sst-2": ['. ', '? ', '! ', '\n'],
    "cola": ['. ', '? ', '! '],
    "rte": ['. '],
    "mrpc": ['. '],
}

# Generated sequences containing bad tokens will be discarded
bad_tokens_mapping = {
    "mnli": ['\n'],
    "qqp": ['\n'],
    "qnli": ['?', '\n'],
    "sst-2": ['"', '“', '”', '\n'],
    "cola": ['"', '“', '”', '\n'],
    "rte": ['\n'],
    "mrpc": ['\n'],
}

# Prompts used by different tasks
# Multiple prompts are included in a list
# Prompts applied to both the sampled text and the generated text are included in a tuple
prompt_mapping = {
    "mnli": {
        "entailment": "In other words,",
        "neutral": "Furthermore,",
        "contradiction": ("There is a rumor that", "However, the truth is:"),
    },
    "qqp": {
        "0": "Furthermore,",
        "1": "In other words,",
    },
    "qnli": {
        "entailment": "",
        "not_entailment": "...",
    },
    "sst-2": {
        "0": "Rating: 1.0",
        "1": "Rating: 5.0",
    },
    "cola": {
        "0": "",
        "1": "",
    },
    "rte": {
        "entailment": "In other words,",
        "not_entailment": "Furthermore,",
    },
    "mrpc": {
        "entailment": "In other words,",
        "not_entailment": "Furthermore,",
    },
}

# repetition reward/penalty parameters
repetition_mapping = {
    "mnli": {
        "entailment": [0.8, 1.1],
        "neutral": [1.3, 1.3],
        "contradiction": [1.1, 1.1],
    },
    "qqp": {
        "0": [1.2, 1.2],
        "1": [1.0, 1.2],
    },
    "qnli": {
        "entailment": [0.9, 1.2],
        "neutral": [0.9, 1.2],
    },
    "sst-2": {
        "0": [1.2],
        "1": [1.2],
    },
    "cola": {
        "0": [1.2],
        "1": [1.2],
    },
    "rte": {
        "entailment": [0.8, 1.1],
        "not_entailment": [1.1, 1.1],
    },
    "mrpc": {
        "entailment": [0.8, 1.1],
        "not_entailment": [1.1, 1.1],
    },
}

# If specified, the stop token leading to the longest sequence (instead of the shortest by default) will be used to terminate a sequence
find_last_stop_token = {
    "sst-2"
}

# If specified, use different temperature values when generating sequences
vary_temperature = {
    "cola": [0.1, 10]
}

# If specified, the remaining generated sequence (after one stop token) will be used to sample another sequence
extract_remaining = {
    "qnli"
}

# If specified, allow the generated sequence to start with "\n" (otherwise, generated sequences starting with "\n" will be discarded)
allow_start_new_line = {
    "qnli"
}


class SuperGenGenerator():

    def __init__(self, args):
        self.args = args
        self.tokenizer = CTRLTokenizer.from_pretrained(args.model_name_or_path)
        self.model = CTRLLMHeadModelWithRepReward.from_pretrained(args.model_name_or_path)
        self.model.to(args.device)
        if args.fp16:
            self.model.half()
        self.set_seed(args.seed)
        self.task_type = task_type_mapping[args.task]
        self.stop_tokens = stop_tokens_mapping[args.task]
        self.control_code = control_code_mapping[args.task]
        self.prompt = prompt_mapping[args.task][args.label]
        self.repetition = repetition_mapping[args.task][args.label]
        self.bad_tokens = bad_tokens_mapping[args.task]
        self.find_stop_idx = self.find_last_stop_idx if args.task in find_last_stop_token else self.find_first_stop_idx
        self.fix_start = fix_start_mapping[args.task] if args.task in fix_start_mapping else None
        self.extract_remain = args.task in extract_remaining
        self.allow_new_line = args.task in allow_start_new_line
        if self.extract_remain:
            for label in prompt_mapping[args.task]:
                if prompt_mapping[args.task][label] == "...":
                    self.remain_label = label
                else:
                    self.prompt = prompt_mapping[args.task][label]
                    self.prompt_label = label
        if self.task_type == "pair":
            assert args.temperature == 0
            assert args.pretrain_corpus_dir is not None
            self.repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else self.repetition[1]
            self.repetition_reward = args.repetition_reward if args.repetition_reward is not None else self.repetition[0]
            f = open(args.pretrain_corpus_dir)
            texts = f.readlines()
            texts = [text.strip() for text in texts]
            chosen_idx = np.random.choice(len(texts), args.num_gen, replace=False)
            self.sampled_texts = [texts[i] for i in chosen_idx]
        else:
            if args.task not in vary_temperature:
                assert args.temperature > 0
            self.repetition_penalty = args.repetition_penalty if args.repetition_penalty is not None else self.repetition[0]
            self.repetition_reward = None
            self.sampled_texts = None
        self.prompt_list = self.prompt if type(self.prompt) == list else [self.prompt]
        if args.task in vary_temperature:
            if type(args.temperature) != list:
                self.temp = vary_temperature[args.task]
            else:
                self.temp = args.temperature
            self.do_sample = True
        elif args.temperature == 0:
            self.temp = 1
            self.do_sample = False
        else:
            self.temp = args.temperature
            self.do_sample = True

    def set_seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.args.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def prepare_input(self, prompt_text):
        encoded_prompt = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        if not any(encoded_prompt[0] == x for x in self.tokenizer.control_codes.values()):
            logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")
        return prompt_text

    def find_first_stop_idx(self, text, skip_len, stop_tokens):
        text = text[skip_len:]
        final_stop_idx = len(text)
        for stop_token in stop_tokens:
            stop_idx = text.find(stop_token)
            if stop_idx != -1:
                if stop_token == '. ':
                    if not (text[stop_idx+len(stop_token)].isupper() or text[stop_idx+len(stop_token)] == '\n'):
                        stop_idx = self.find_first_stop_idx(text, stop_idx+len(stop_token), stop_tokens)
                if stop_idx < final_stop_idx and stop_idx != -1:
                    final_stop_idx = stop_idx
                    stop_token_len = len(stop_token)
        if final_stop_idx < len(text):
            final_stop_idx += skip_len + stop_token_len - 1
        else:
            final_stop_idx = -1
        return final_stop_idx

    def find_last_stop_idx(self, text, skip_len, stop_tokens):
        text = text[skip_len:]
        final_stop_idx = -1
        for stop_token in stop_tokens:
            stop_idx = text.find(stop_token)
            if stop_idx != -1:
                if stop_token == '. ':
                    if not (text[stop_idx+len(stop_token)].isupper() or text[stop_idx+len(stop_token)] == '\n'):
                        stop_idx = self.find_last_stop_idx(text, stop_idx+len(stop_token), stop_tokens)
                if stop_idx > final_stop_idx and stop_idx != -1:
                    final_stop_idx = stop_idx
                    stop_token_len = len(stop_token)
        if final_stop_idx > 0:
            final_stop_idx += skip_len + stop_token_len - 1
        else:
            final_stop_idx = -1
        return final_stop_idx

    def generate_one(self, seed, sample_text=None):
        self.set_seed(seed)

        # always start with control codes (when generator is CTRL)
        start = self.control_code + ' '
        choice_idx = np.random.choice(len(self.prompt_list), 1)
        prompt = self.prompt_list[choice_idx[0]]
        if type(prompt) == tuple:
            assert len(prompt) == 2 and sample_text is not None
            start_prompt = prompt[0]
            conj_prompt = prompt[1]
            lowercase_sampled = True
        else:
            start_prompt = prompt if sample_text is None else None
            conj_prompt = None if sample_text is None else prompt
            lowercase_sampled = False
        
        # append start prompt if any
        if start_prompt is not None and len(start_prompt) > 0:
            start += start_prompt + ' '
        prompt_text = start

        # append sample text if any
        if sample_text is not None:
            orig_sample_text = sample_text
            if lowercase_sampled:
                sample_text = orig_sample_text[0].lower() + orig_sample_text[1:]
            else:
                sample_text = orig_sample_text
            start += sample_text + ' '
            first_sent_text = start
            preprocessed_prompt_text = self.prepare_input(prompt_text)
            preprocessed_first_sent_text = self.prepare_input(first_sent_text)
            encoded_prompt = self.tokenizer.encode(
                preprocessed_prompt_text, add_special_tokens=False, return_tensors="pt",
            )
            encoded_first_sent = self.tokenizer.encode(
                preprocessed_first_sent_text, add_special_tokens=False, return_tensors="pt",
            )
            reward_span = torch.tensor([len(encoded_prompt[0]), len(encoded_first_sent[0])])
        else:
            reward_span = None
        
        # append conjunction prompt if any
        if conj_prompt is not None and len(conj_prompt) > 0:
            start += conj_prompt + ' '
        
        # append fixed start tokens if any
        if self.fix_start is not None:
            choice_idx = np.random.choice(len(self.fix_start), 1)
            start_words = self.fix_start[choice_idx[0]]
            start += start_words + ' '
        else:
            start_words = None
        
        preprocessed_start_text = self.prepare_input(start)
        encoded_start = self.tokenizer.encode(
            preprocessed_start_text, add_special_tokens=False, return_tensors="pt",
        )
        encoded_start = encoded_start.to(self.args.device)
        if encoded_start.size()[-1] == 0:
            input_ids = None
        else:
            input_ids = encoded_start
        if sample_text is not None:
            max_len = len(input_ids[0]) + self.args.max_len
            if len(input_ids[0]) > 1.5 * self.args.max_len:
                return None
        else:
            max_len = self.args.max_len
        if type(self.temp) == list:
            choice_idx = np.random.choice(len(self.temp), 1)
            temp = float(self.temp[choice_idx[0]])
        else:
            temp = self.temp
        outputs = self.model.generate(
            input_ids=input_ids,
            reward_span=reward_span,
            max_length=max_len,
            temperature=temp,
            top_k=self.args.k,
            top_p=self.args.p,
            repetition_penalty=self.repetition_penalty,
            repetition_reward=self.repetition_reward,
            do_sample=self.do_sample,
            num_return_sequences=1,
            output_scores=True,
            return_dict_in_generate=True,
        )
        output_sequences = outputs["sequences"][0]
        tokens = [self.tokenizer.convert_ids_to_tokens(wid.item()) for wid in output_sequences]
        scores = outputs["scores"]

        generated_sequence = output_sequences
        generated_sequence = generated_sequence.tolist()

        # Decode text
        text = self.tokenizer.decode(generated_sequence, clean_up_tokenization_spaces=True)
        
        start_len = len(self.tokenizer.decode(encoded_start[0], clean_up_tokenization_spaces=True))

        if not self.allow_new_line and (text[start_len:].startswith("\n") or text[start_len:].startswith(" \n")):
            return None
        
        skip_len = len(start)
        if len(self.stop_tokens) > 0:
            final_stop_idx = self.find_stop_idx(text, skip_len, self.stop_tokens)
            # Remove all text after the stop token
            trunc_text = text[:final_stop_idx]
            if self.extract_remain:
                remain_text = text[final_stop_idx:]
                sents = sent_tokenize(remain_text.strip())
                if len(sents) > 1:
                    select_idx = np.random.choice(len(sents)-1, 1)
                    remain_text = sents[select_idx[0]]
                    extra_sequence = remain_text.strip()
                else:
                    return None
        if final_stop_idx == -1:
            return None

        total_sequence = (trunc_text[start_len:])
        total_sequence = total_sequence.strip()
        for bad_token in self.bad_tokens:
            if bad_token in total_sequence:
                return None
            if self.extract_remain and bad_token in extra_sequence:
                return None
        start_idx = len(input_ids[0])
        num_skip = 0
        if self.allow_new_line:
            while tokens[start_idx] == '\n':
                num_skip += 1
                start_idx += 1
        assert total_sequence.startswith(tokens[start_idx].split('@@')[0]), f"total_sequence: {total_sequence}; start_token: {tokens[start_idx]}"
        total_sequence_split = total_sequence.split(' ')
        j = 0
        subtoken = ''
        valid_flag = True
        for i, token in enumerate(tokens[start_idx:]):
            if j == len(total_sequence_split):
                break
            if subtoken + token != total_sequence_split[j]:
                try:
                    assert token.endswith('@@') or total_sequence_split[j][-1] in string.punctuation
                except AssertionError:
                    valid_flag = False
                    break
                subtoken += token.split('@@')[0]
            else:
                subtoken = ''
                j += 1
        if valid_flag == False:
            return None

        with torch.no_grad():
            scores = scores[num_skip:num_skip+i]
            scores = torch.cat(scores, dim=0) * temp
            token_ids = output_sequences[start_idx:i+start_idx]
            probs = F.log_softmax(scores, dim=-1)
            token_probs = probs.gather(dim=-1, index=token_ids.unsqueeze(-1)).mean()
        if start_words is not None:
            gen_text = start_words + ' ' + total_sequence
        else:
            gen_text = total_sequence[0].upper() + total_sequence[1:]
        if sample_text is not None:
            res = {"text1": orig_sample_text, 
                   "text2": gen_text, 
                   "label": self.args.label,
                   "start_prompt": prompt_text, 
                   "conj_prompt": conj_prompt, 
                   "score": token_probs.item()}
            if self.args.print_res:
                print(res)
        else:
            res = {"text": gen_text, 
                   "label": self.args.label,
                   "start_prompt": prompt_text,
                   "score": token_probs.item()}
            if self.args.print_res:
                print(res)
        if self.extract_remain:
            res["extra"] = extra_sequence
        return res

    def save_res(self, gen_res):
        os.makedirs(self.args.save_dir, exist_ok=True)
        if self.extract_remain:
            gen_prompt_res = []
            gen_extra_res = []
            for res in gen_res:
                prompt_res = {k: v for k, v in res.items() if k != "extra"}
                prompt_res["label"] = self.prompt_label
                gen_prompt_res.append(prompt_res)
                extra_res = {k: v for k, v in res.items() if k != "extra"}
                extra_res["label"] = self.remain_label
                extra_res["text2"] = res["extra"]
                gen_extra_res.append(extra_res)
            save_name = os.path.join(self.args.save_dir, f"{self.args.task}_{self.prompt_label}_{self.args.num_gen}")
            with open(f"{save_name}.json", 'w') as f:
                res = json.dumps(gen_prompt_res)
                f.write(res)
                f.close()
            new_dict = sort_score(f"{save_name}.json")
            save(f"{save_name}_sorted.json", new_dict)
            print(f"saved to {save_name}_sorted.json")
            save_name = os.path.join(self.args.save_dir, f"{self.args.task}_{self.remain_label}_{self.args.num_gen}")
            with open(f"{save_name}.json", 'w') as f:
                res = json.dumps(gen_extra_res)
                f.write(res)
                f.close()
            new_dict = sort_score(f"{save_name}.json")
            save(f"{save_name}_sorted.json", new_dict)
            print(f"saved to {save_name}_sorted.json")
        else:
            save_name = os.path.join(self.args.save_dir, f"{self.args.task}_{self.args.label}_{self.args.num_gen}")
            with open(f"{save_name}.json", 'w') as f:
                res = json.dumps(gen_res)
                f.write(res)
                f.close()
            new_dict = sort_score(f"{save_name}.json")
            save(f"{save_name}_sorted.json", new_dict)
            print(f"saved to {save_name}_sorted.json")

    def generate_all(self):
        gen_res = []
        for seed in tqdm(range(self.args.num_gen)):
            if self.sampled_texts is None:
                sample_text = None
            else:
                sample_text = self.sampled_texts[seed]
            res = self.generate_one(seed, sample_text)
            if res is not None:
                gen_res.append(res)
        self.save_res(gen_res)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_corpus_dir', default=None,)
    parser.add_argument('--task', default='mnli',)
    parser.add_argument('--label', default='entailment',)
    parser.add_argument('--model_type', default='ctrl',)
    parser.add_argument('--model_name_or_path', default='ctrl',)
    parser.add_argument('--temperature', default='0.2')
    parser.add_argument('--repetition_reward', default=None, type=float)
    parser.add_argument('--repetition_penalty', default=None, type=float)
    parser.add_argument('--p', default=1.0, type=float)
    parser.add_argument('--k', default=10, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--no_cuda', default=False,)
    parser.add_argument('--fp16', default=False,)
    parser.add_argument('--num_gen', default=10, type=int)
    parser.add_argument('--max_len', default=60, type=int)
    parser.add_argument('--save_dir', default='temp_gen')
    parser.add_argument('--print_res', action='store_true')
    args = parser.parse_args()
    print(args)
    args.task = args.task.lower()
    args.temperature = eval(args.temperature)
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")

    # Generate texts for all labels
    if args.label == "all":
        for label in prompt_mapping[args.task]:
            args.label = label
            generator = SuperGenGenerator(args)
            generator.generate_all()
            # If texts of all labels are generated in one pass 
            # (by varying temperatures or extracting from the same generated text),
            # no need to redo generation for each label
            if args.task in vary_temperature or args.task in extract_remaining:
                break
    else:
        generator = SuperGenGenerator(args)
        generator.generate_all()


if __name__ == "__main__":
    main()
    