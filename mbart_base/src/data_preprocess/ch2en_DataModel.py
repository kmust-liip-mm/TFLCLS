from functools import reduce

import numpy
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from transformers import AutoTokenizer
from data_preprocess.MSMO_Dataset import MSMODataset
from utils.utils import get_mask, pad_sents


class SummaryDataModule1(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, src_lang=args.src_lang,
                                                       tgt_lang=args.tgt_lang)

        data_files = {}
        if self.args.ch2en_mode == 'translate':
            if self.args.mode == 'train':
                data_files['train'] = 'dataset/ch2en/train_zh2en_tra_dataset.csv'
                data_files['val'] = 'dataset/ch2en/val_zh2en_tra_dataset.csv'
            data_files['test'] = 'dataset/ch2en/test_zh2en_tra_dataset.csv'
        elif self.args.ch2en_mode == 'cls_summary':
            if self.args.mode == 'train':
                data_files['train'] = 'dataset/ch2en/train_zh2en_sum_dataset.csv'
                data_files['val'] = 'dataset/ch2en/val_zh2en_sum_dataset.csv'
            data_files['test'] = 'dataset/ch2en/test_zh2en_sum_dataset.csv'
        elif self.args.ch2en_mode == 'summary':
            if self.args.mode == 'train':
                data_files['train'] = 'dataset/ch2en/train_zh2zh_sum_dataset.csv'
                data_files['val'] = 'dataset/ch2en/val_zh2zh_sum_dataset.csv'
            data_files['test'] = 'dataset/ch2en/test_zh2zh_sum_dataset.csv'
        elif self.args.ch2en_mode == 'dd_summary':
            if self.args.mode == 'train':
                data_files['train'] = 'dataset/ch2en/train_zh2enzh_sum_dataset.csv'
                data_files['val'] = 'dataset/ch2en/val_zh2enzh_sum_dataset.csv'
            data_files['test'] = 'dataset/ch2en/test_zh2enzh_sum_dataset.csv'
        raw_datasets = load_dataset('csv', data_files=data_files, )
        # split Dataset
        if self.args.mode == 'train':
            if args.train_dataset_length != -1:
                raw_datasets["train"] = raw_datasets["train"].select(range(args.train_dataset_length))
            if args.val_dataset_length != -1:
                raw_datasets["val"] = raw_datasets["val"].select(range(args.val_dataset_length))
        if args.test_dataset_length != -1:
            raw_datasets["test"] = raw_datasets["test"].select(range(args.test_dataset_length))
        # Tokenize
        if self.args.mode == 'train':
            self.train_loader = DataLoader(dataset=raw_datasets["train"], \
                                           batch_size=self.args.batch_size, \
                                           num_workers=args.num_workers, \
                                           shuffle=True, \
                                           collate_fn=self.collate_fn)
            self.val_loader = DataLoader(dataset=raw_datasets["val"], \
                                         batch_size=self.args.val_batch_size, \
                                         num_workers=args.num_workers, \
                                         collate_fn=self.collate_fn)
        self.test_loader = DataLoader(dataset=raw_datasets["test"], \
                                      batch_size=self.args.test_batch_size, \
                                      num_workers=args.num_workers, \
                                      collate_fn=self.collate_fn)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def collate_fn(self, data):
        if self.args.is_prompt:
            articles = [self.tokenizer.unk_token * self.args.prompt_token_num + d['src'] for d in data]
        else:
            articles = [d['src'] for d in data]
            if self.args.ch2en_mode == 'summary':
                refers = [d['tgt'] for d in data]
            elif self.args.ch2en_mode == 'dd_summary':
                refers = [d['tgt'].split('O.o')[0].lower() for d in data]
                zh_refers = [d['tgt'].split('O.o')[1]for d in data]
            else:
                refers = [d['tgt'].lower() for d in data]
        if self.args.is_peft_prompt:
            model_inputs = self.tokenizer.batch_encode_plus(articles,
                                                            max_length=self.args.sent_token_len - self.args.prompt_token_num,
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors=self.args.return_tensors)
            # Tokenize targets with the `text_target` keyword argument
            labels = self.tokenizer.batch_encode_plus(refers,
                                                      max_length=self.args.ref_token_len,
                                                      # +self.args.prompt_token_num,
                                                      padding=self.args.padding,
                                                      truncation=self.args.truncation,
                                                      return_tensors=self.args.return_tensors)
        else:
            model_inputs = self.tokenizer(articles,
                                            text_target=refers,
                                            max_length=self.args.sent_token_len,
                                            # -self.args.prompt_token_num,
                                            padding=True,
                                            truncation=self.args.truncation,
                                            return_tensors=self.args.return_tensors)


            if self.args.ch2en_mode == 'dd_summary':
                zh_labels = self.tokenizer(zh_refers,
                                              max_length=self.args.ref_token_len,
                                              # -self.args.prompt_token_num,
                                              padding=True,
                                              truncation=self.args.truncation,
                                              return_tensors=self.args.return_tensors)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if self.args.padding == "max_length" and self.args.ignore_pad_token_for_loss:
            for batch_i in range(model_inputs['labels'].size()[0]):
                for token_i in range(model_inputs['labels'].size()[1]):
                    if model_inputs['labels'][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                        model_inputs['labels'][batch_i][token_i] = -100

        if self.args.ch2en_mode == 'dd_summary':
            if self.args.padding == "max_length" and self.args.ignore_pad_token_for_loss:
                for batch_i in range(zh_labels["input_ids"].size()[0]):
                    for token_i in range(zh_labels["input_ids"].size()[1]):
                        if zh_labels["input_ids"][batch_i][token_i].item() == self.tokenizer.pad_token_id:
                            zh_labels["input_ids"][batch_i][token_i] = -100

        if self.args.ch2en_mode == 'dd_summary':
            return {'tokenized_contents': model_inputs,
                    'refers': refers,
                    'zh_tokenized_refers': zh_labels,
                    'zh_refers': zh_refers,
                    }
        else:
            return {'tokenized_contents': model_inputs,
                    'refers': refers,
                    }