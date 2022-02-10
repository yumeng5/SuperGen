import json
import numpy as np
import argparse
import os
from collections import defaultdict


task_label = {
    "mnli": ["entailment", "neutral", "contradiction"],
    "qqp": ["0", "1"],
    "qnli": ["entailment", "not_entailment"],
    "sst-2": ["0", "1"],
    "cola": ["0", "1"],
    "rte": ["entailment", "not_entailment"],
    "mrpc": ["entailment", "not_entailment"],
}

# If specified, do not select top-ranked texts for that label; select randomly instead
no_sort = {
    "mnli": ["neutral"]
}

# If specified, use different temperature values when generating sequences; need to assign labels based on generation scores
vary_temperature = {
    "cola"
}


def read_files(read_dir, task):
    for (_, _, filenames) in os.walk(read_dir):
        break
    file_dict = {}
    if task in vary_temperature:
        found = False
        for f in filenames:
            if f.startswith(f"{task}") and f.endswith("_sorted.json"):
                if found:
                    print(f"Found more than one sorted generated file for task {task}! Make sure there is only one!")
                    exit(-1)
                found = True
                file_dict["all"] = os.path.join(read_dir, f)
        return file_dict
    for label in task_label[task]:
        found = False
        for f in filenames:
            if f.startswith(f"{task}_{label}") and f.endswith("_sorted.json"):
                if found:
                    print(f"Found more than one sorted generated file for task {task}, label {label}! Make sure there is only one!")
                    exit(-1)
                found = True
                file_dict[label] = os.path.join(read_dir, f)
        if not found:
            print(f"Not found sorted generated file for task {task}, label {label}!")
            exit(-1)
    return file_dict


def combine(task, gen_file_dict, k=None):
    combined_dict = []
    data_dicts = []
    if task in vary_temperature:
        assert len(gen_file_dict) == 1
        data_dict = json.load(open(gen_file_dict["all"], 'r'))
        print(f"{len(data_dict)} total samples")
        if k is None:
            k = int(len(data_dict)/2)
        pos_dict = data_dict[:k]
        for data in pos_dict:
            data["label"] = "1"
            combined_dict.append(data)
        neg_dict = data_dict[-k:]
        for data in neg_dict:
            data["label"] = "0"
            combined_dict.append(data)
        print(f"Label 0: {len(neg_dict)} selected samples")
        print(f"Label 1: {len(pos_dict)} selected samples")
    else:
        for label, file_dir in gen_file_dict.items():
            data_dict = json.load(open(file_dir, 'r'))
            if task in no_sort and label in no_sort[task]:
                data_dict = np.random.permutation(data_dict)
            print(f"Label {label}: {len(data_dict)} total samples")
            data_dicts.append(data_dict)
        if k is None:
            k = max([len(data_dict) for data_dict in data_dicts])
        label_count = defaultdict(int)
        for i in range(k):
            for data_dict in data_dicts:
                if i < len(data_dict):
                    combined_dict.append(data_dict[i])
                    label_count[data_dict[i]["label"]] += 1
        for label in label_count:
            print(f"Label {label}: {label_count[label]} selected samples")
    print(f"Total {len(combined_dict)} samples")
    return combined_dict


# Sort generated text by average log probability score
def sort_score(gen_file_dir):
    data_dict = json.load(open(gen_file_dir, 'r'))
    text_set = []
    new_data_dict = []
    for data in data_dict:
        text = data["text"] if "text" in data else data["text1"]
        if text not in text_set:
            new_data_dict.append(data)
        text_set.append(text)
    data_dict = new_data_dict
    scores = np.array([data["score"] for data in data_dict])
    sort_idx = np.argsort(-scores)
    new_dict = []
    for i in range(len(sort_idx)):
        new_dict.append(data_dict[sort_idx[i]])
    return new_dict


def save(save_file_dir, save_dict):
    with open(save_file_dir, 'w') as f:
        res = json.dumps(save_dict)
        f.write(res)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read_dir', default='temp_gen')
    parser.add_argument('--save_dir', default='data/MNLI')
    parser.add_argument('--task', default='mnli',)
    parser.add_argument('--num_select_samples', default=6000, type=int)
    args = parser.parse_args()
    task = args.task.lower()
    k = int(args.num_select_samples/len(task_label[task]))
    gen_file_dict = read_files(args.read_dir, task)
    combined_dict = combine(task, gen_file_dict, k)
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = os.path.join(args.save_dir, "train.json")
    save(save_name, combined_dict)
    print(f"saved to {save_name}")

if __name__ == "__main__":
    main()
