from functools import reduce

import numpy
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from data_preprocess.MSMO_Dataset import MSMODataset
from utils.utils import get_mask, pad_sents


class SummaryDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, src_lang=args.src_lang, tgt_lang=args.tgt_lang)
        data_files = {}

        if self.args.mode == 'train':
            data_files["train"] = self.args.dataset_path + 'train.csv'
            data_files["val"] = self.args.dataset_path + 'val.csv'
        data_files["test"] = self.args.dataset_path + 'test.csv'
        raw_datasets = load_dataset('csv', data_files=data_files, cache_dir=None)
        # split Dataset
        if self.args.mode == 'train':
            if args.train_dataset_length != -1:
                raw_datasets["train"] = raw_datasets["train"].select(range(args.train_dataset_length))
            if args.val_dataset_length != -1:
                raw_datasets["val"] = raw_datasets["val"].select(range(args.val_dataset_length))
        if args.test_dataset_length != -1:
            raw_datasets["test"] = raw_datasets["test"].select(range(args.test_dataset_length))
        if self.args.mode == 'train':
            self.train_loader = DataLoader(dataset=raw_datasets["train"], \
                                           batch_size=self.args.batch_size, \
                                           num_workers=args.num_workers, \
                                           shuffle=False, \
                                           collate_fn=self.collate_fn)
            self.val_loader = DataLoader(dataset=raw_datasets["val"], \
                                         batch_size=self.args.val_batch_size, \
                                         num_workers=args.num_workers, \
                                         shuffle=False, \
                                         collate_fn=self.collate_fn)
        self.test_loader = DataLoader(dataset=raw_datasets["test"], \
                                      batch_size=self.args.test_batch_size, \
                                      num_workers=args.num_workers, \
                                      shuffle=False, \
                                      collate_fn=self.collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def collate_fn(self, data):
        if self.args.is_prompt:
            articles = [self.tokenizer.unk_token * self.args.prompt_token_num + d['article'] for d in data]
        else:
            articles = [d['article'] for d in data]
        if self.args.is_peft_prompt:
            model_inputs = self.tokenizer(articles,
                                          max_length=self.args.sent_token_len-self.args.prompt_token_num,
                                          padding=self.args.padding,
                                          truncation=self.args.truncation,
                                          return_tensors=self.args.return_tensors)
            # Tokenize targets with the `text_target` keyword argument
            labels = self.tokenizer([d['summary'] for d in data],
                                    max_length=self.args.ref_token_len,#+self.args.prompt_token_num,
                                    padding=self.args.padding,
                                    truncation=self.args.truncation,
                                    return_tensors=self.args.return_tensors)
        else:
            model_inputs = self.tokenizer(articles,
                                          max_length=self.args.sent_token_len,#-self.args.prompt_token_num,
                                          padding=self.args.padding,
                                          truncation=self.args.truncation,
                                          return_tensors=self.args.return_tensors)
            # Tokenize targets with the `text_target` keyword argument
            labels = self.tokenizer([d['summary'] for d in data],
                                    max_length=self.args.ref_token_len,#-self.args.prompt_token_num,
                                    padding=self.args.padding,
                                    truncation=self.args.truncation,
                                    return_tensors=self.args.return_tensors)
        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.args.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            for batch_i in range(labels["input_ids"].size()[0]):
                for token_i in range(labels["input_ids"].size()[1]):
                    if labels["input_ids"][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                        labels["input_ids"][batch_i][token_i] = -100

        return {'tokenized_contents': model_inputs,
                'tokenized_refers': labels,
                'refers': [d['summary'] for d in data],}
