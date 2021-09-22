import json
import torch
from t5multilabelModel import T5FineTuner
from transformers import (T5Tokenizer)
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
import textwrap
with open('nolabeldata.json', 'r') as f:
    data = json.load(f)


#sample_text = [data['657'], data['4376'], data['8934'], data['332'], data['7']]
model_path = './output/model.ckpt'
model = T5FineTuner.load_from_checkpoint(model_path, map_location='cuda:0')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

for k, d in data.items():
    text = 'multi classification: ' + d['data']['title'] + " "+ d['data']['content']
    text = text.replace('\n', '')
    token_id = tokenizer.encode_plus(text, max_length=len(text), truncation=True, return_tensors="pt").input_ids
    #input_id = {"source_ids": token_id['input_ids']}
    pred = model.model.generate(token_id)
    pred = tokenizer.batch_decode(pred, skip_special_tokens=True)
    text = textwrap.fill(text, width=80)
    print("==========Text============")
    print(text)
    print("==========Pred============")
    print(pred)
