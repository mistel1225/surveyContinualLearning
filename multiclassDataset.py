import json
import os
from torch.utils.data import Dataset
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from transformers import (T5Tokenizer)
import pandas as pd
class multiclassDataset():
    def __init__(self, tokenizer, data_dir, label_dir, max_len=512):
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.inputs = []
        self.targets = []
        self._build()
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = self.inputs[index]["attention_mask"].squeeze()
        target_mask = self.targets[index]["attention_mask"].squeeze()
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
    def _build(self):
        self._build_examples_from_files()
    def _build_examples_from_files(self):
        data = pd.read_fwf(self.data_dir, header=None)
        label = pd.read_fwf(self.label_dir, header=None)
        mapping = pd.read_csv('tweeteval/datasets/emoji/mapping.txt', header=None, delimiter='\t')
        mapping_dict = mapping.to_dict('index')
        mapping_dict = {val[2]:val[0] for val in mapping_dict.values()}
        self.emoji_idx_mapping = mapping_dict
        data = data[0]
        label = label[0]
        for d, l in tqdm(zip(data, label), total=data.size):
            _ = "multi-class classification: "+d
            tokenizer_inputs = self.tokenizer.batch_encode_plus(
                [_], max_length = self.max_len, padding='max_length',
                truncation=True, return_tensors="pt"
                    )
            self.inputs.append(tokenizer_inputs)
            _ = mapping[2][l]
            tokenizer_targets = self.tokenizer.batch_encode_plus(
                [_], max_length=32, padding='max_length', truncation=True, return_tensors="pt"
                    )
            self.targets.append(tokenizer_targets)
def get_dataset(tokenizer, data_dir, label_dir, args):
    return multiclassDataset(tokenizer=tokenizer, data_dir=data_dir, label_dir=label_dir, max_len=args.max_seq_length)

