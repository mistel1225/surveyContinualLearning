import torch
import pytorch_lightning as pl
import argparse
from t5multilabelModel import T5FineTuner
from t5multilabelDataset import get_dataset
from multilabelTrain import args_dict
from torch.utils.data import Dataset, DataLoader
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import load_metric
import sys
sys.path.append("../../utils/")
from compute_metrics import metrics
from tqdm import tqdm
metric = metrics()

args_dict['model_name_or_path'] = './output/model.ckpt'
#args_dict['eval_batch_size'] = 8
args = argparse.Namespace(**args_dict)
#model = T5ForConditionalGeneration.from_pretrained(torch.load(args.model_name_or_path)).cuda()
model = T5FineTuner.load_from_checkpoint(args_dict['model_name_or_path'], map_location = 'cuda:0')
tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
dataloader = model.val_dataloader()
for d in tqdm(dataloader):
    pred = model.generate(d)

    gt = tokenizer.batch_decode(d['target_ids'], skip_special_tokens=True)
    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
        
    ground_truth = [_.split(',') for _ in gt]
    prediction = [_.split(',') for _ in pred]
    print(ground_truth)
    print(prediction)
    content = tokenizer.batch_decode(d['source_ids'], skip_special_tokens=True)
    
    metric.add_batch(prediction, ground_truth)
    '''
    for content, predict in zip(data, output):
        print("*******context*******\n{0}\n********predict*******\n{1}".format(content, predict))
    '''
metric.compute_score()


