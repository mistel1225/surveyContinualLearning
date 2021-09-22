import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from multiclassDataset import get_dataset
from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
        )
from sklearn.metrics import f1_score, accuracy_score
class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.model = T5ForConditionalGeneration.from_pretrained(self.hparams.model_name_or_path)
        self.tokenizer = T5Tokenizer.from_pretrained(self.hparams.tokenizer_name_or_path)
    def forward(self, batch):
        return self.model.generate(batch["source_ids"])
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        #preprocess and postprocess here
        return self(batch)
    def _step(self, batch):
        labels = batch["target_ids"] 
        #labels[labels[:,:]==self.tokenizer.pad_token_id] = -100
        outputs = self.model(
            input_ids=batch['source_ids'],
            attention_mask=batch["source_mask"],
            labels=labels,
            decoder_attention_mask=batch["target_mask"]
        )     
        loss = outputs.loss
        return loss
    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('train_loss', loss, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log('val_loss', loss, prog_bar=False, logger=True)
        output = self(batch)
        return {'pred': output, 'gt': batch['target_ids']}
    def validation_epoch_end(self, val_step_outputs):
        #print(val_step_outputs)
        gt_list = []
        pred_list = []
        emoji_idx_mapping = self.val_dataset.emoji_idx_mapping
        for out in val_step_outputs:
            pred = out['pred']
            gt = out['gt']
            gt = self.tokenizer.batch_decode(gt, skip_special_tokens=True)
            pred = self.tokenizer.batch_decode(pred, skip_special_tokens=True)
            for l1, l2 in zip(pred, gt):
                try:
                    pred_idx = emoji_idx_mapping[l1]
                    gt_idx = emoji_idx_mapping[l2]
                    pred_list.append(pred_idx)
                    gt_list.append(gt_idx)
                except:
                    gt_list.append(emoji_idx_mapping[l2])
                    pred_list.append(-1)
        acc = accuracy_score(gt_list, pred_list)
        f1 = f1_score(gt_list, pred_list, average='macro')
        self.log('val_acc', acc, prog_bar=False, logger=True)
        self.log('val_f1_score', f1, prog_bar=False, logger=True)
        #print("acc: {}".format(acc))
        #print("f1 macro: {}".format(f1))
            

    '''
    def test_step(self, batch, batch_idx):
        pass
    def test_step_end(test_outputs):
        pass
    '''
    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_ground_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.hparams.weight_decay,
                    },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                    },
                ]
        optimizer = AdamW(optimizer_ground_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        t_total = (
                ((self.hparams.num_train_size) // (self.hparams.train_batch_size) // (self.hparams.gradient_accumulation_steps)) 
                * float(self.hparams.num_train_epochs)
                )
        scheduler = get_linear_schedule_with_warmup(self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total)
        self.lr_scheduler = scheduler
        return [optimizer], [scheduler]
    '''
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None, on_tpu=None, using_native_amp=None, using_lbfgs=None, optimizer_clousure=None, ):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()
    '''
    '''
    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr":self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict
    '''
    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.train_dir, label_dir = self.hparams.train_label_dir, args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=False, shuffle=True, 
                num_workers=8)
        '''
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size) // self.hparams.gradient_accumulation_steps)
                * float(self.hparams.num_train_epochs)
                )
        self.total_step = t_total
        scheduler = get_linear_schedule_with_warmup(
                self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
                )
        self.lr_scheduler = scheduler
        '''
        self.train_dataset = train_dataset
        return dataloader
    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, data_dir=self.hparams.val_dir, label_dir=self.hparams.val_label_dir, args=self.hparams)
        self.val_dataset = val_dataset
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=8)
