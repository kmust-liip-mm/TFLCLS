import argparse

import torch
from pytorch_lightning.callbacks import LearningRateFinder
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MBartModel, AutoConfig

from models.dd_modeling_mbart import MBartForConditionalGeneration
from models.model_interface import BaseModel
from models.modeling_metrics import ROUGEScore
from models.soft_embedding import SoftEmbedding
import pytorch_lightning as pl
from peft import get_peft_model, LoraConfig, TaskType, PromptTuningConfig, PrefixTuningConfig
from utils.AutomaticWeightedLoss import AutomaticWeightedLoss, AutomaticWeightedOneLoss, AutomaticWeightedTwoLoss
from utils.NCLS_metrics.metric import calculate_rouge, calculate_bleu
from utils.loss_handler import MultiLossLayer

T = 2.0

class dd_peft_MBart(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        if type(args) == dict:
            args = argparse.Namespace(**args)
        self.args = args
        self.learning_rate = args.learning_rate

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                                       src_lang=args.src_lang,
                                                       tgt_lang=args.tgt_lang,)
        if args.is_dd:
            self.model = MBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
            self._init_dd_decoder_with_decoder_params()
            print('init double decoder ok')
        else:
            self.model = MBartForConditionalGeneration.from_pretrained(args.model_name_or_path)
        if args.is_prompt:
            self.model.model.encoder.embed_tokens = SoftEmbedding(self.model.model.encoder.embed_tokens,
                                                             args.prompt_token_num)
        if args.is_peft_prompt:
            peft_config = PromptTuningConfig(
                peft_type="PROMPT_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=args.prompt_token_num,
                num_transformer_submodules=1,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        if args.is_prefix:
            peft_config = PrefixTuningConfig(
                    peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=args.prompt_token_num,
                    num_transformer_submodules=1
                )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        if args.is_lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                bias=args.lora_bias
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()


        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            self.tokenizer.lang_code_to_id[args.tgt_lang] if args.tgt_lang is not None else None
        )
        self.model.config.forced_bos_token_id = forced_bos_token_id
        self.model.config.decoder_start_token_id = forced_bos_token_id
        self.test_abs_rouge = ROUGEScore()
        self.save_hyperparameters(args)
        self.cost = torch.zeros(2, dtype=torch.float32, requires_grad=False)
        if self.args.loss_type == 46:
            self.atw = AutomaticWeightedLoss(2)
        if self.args.loss_type == 47:
            self.avg_cost = torch.zeros([20, 2], dtype=torch.float32, requires_grad=False)
            self.lambda_weight = torch.ones([2, 20], requires_grad=False)
            self.epoch_num = -1
            self.batch_data_len = 0
        if self.args.loss_type == 48:
            self.atw = AutomaticWeightedOneLoss(ratio=self.args.r)
        if self.args.loss_type == 49:
            self.atw = AutomaticWeightedTwoLoss(2)


    def _init_dd_decoder_with_decoder_params(self):
        for zh_param, decoder_param in zip(self.model.model.base_model.zh_decoder.parameters(),
                                           self.model.model.decoder.parameters()):
            zh_param.data = decoder_param.data.clone()

    def forward(self, input_ids, attention_mask, labels, zh_labels):
        loss = self.model(input_ids=input_ids,
                          attention_mask=attention_mask,
                          labels=labels,
                          zh_labels=zh_labels)[0]
        return loss

    def training_step(self, batch, batch_idx):
        # get loss
        loss = self(**batch['tokenized_contents'], zh_labels=batch['zh_tokenized_refers']['input_ids'])
        if self.args.loss_type == 48 or self.args.loss_type == 49:
            loss = self.atw(*loss)
        if self.args.loss_type == 47:
            self.cost[0] = loss[0].clone().detach()
            self.cost[1] = loss[1].clone().detach()
            self.avg_cost[self.epoch_num, :2] += self.cost[:2] / self.batch_data_len
            loss = sum([self.lambda_weight[i, self.epoch_num].clone() * loss[i] for i in range(2)])
        elif self.args.loss_type == 46:
            loss = self.atw(*loss)
        elif self.args.loss_type == 45:
            loss = MultiLossLayer(loss).get_loss()
        elif self.args.loss_type == 44:
            loss = loss[0]/2 + loss[1]/2
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_start(self) -> None:
        if self.args.loss_type == 47:
            if self.batch_data_len == 0:
                self.batch_data_len = len(self.trainer.train_dataloader)
            self.epoch_num += 1
            self.cost = torch.zeros(2, dtype=torch.float32)
            if self.epoch_num == 0 or self.epoch_num == 1:
                self.lambda_weight[:, self.epoch_num] = 1.0
            else:
                w_1 = self.avg_cost[self.epoch_num - 1, 0] / self.avg_cost[self.epoch_num - 2, 0]
                w_2 = self.avg_cost[self.epoch_num - 1, 1] / self.avg_cost[self.epoch_num - 2, 1]
                self.lambda_weight[0, self.epoch_num] = 2 * torch.exp(w_1 / T) / (torch.exp(w_1 / T) + torch.exp(w_2 / T))
                self.lambda_weight[1, self.epoch_num] = 2 * torch.exp(w_2 / T) / (torch.exp(w_1 / T) + torch.exp(w_2 / T))


    def validation_step(self, batch, batch_idx):
        # batch
        # get summary
        batch['tokenized_contents'].pop('labels')
        summary_ids = self.model.generate(**batch['tokenized_contents'],
                                          num_beams=self.args.num_beams,
                                          max_length=self.args.max_output_len,
                                          early_stopping=True,
                                          no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                          )
        return [summary_ids, batch['refers']]

    def validation_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            try:
                summary_id = item[0]
                if self.args.tgt_lang != "zh_CN":
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]
                    self.test_abs_rouge.update(one_summary, item[1])
                else:
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]

                    self.test_abs_rouge.update([' '.join(self.tokenizer.tokenize(sum))[1:] for sum in one_summary],
                                               [' '.join(self.tokenizer.tokenize(sum))[1:] for sum in item[1]])
                summary += one_summary
                reference += item[1]
            except:
                print("某个生成出错啦")
        test_abs_rouge_results = self.test_abs_rouge.compute()
        self.log('val_R1', test_abs_rouge_results["rouge-1"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_R2', test_abs_rouge_results["rouge-2"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('val_RL', test_abs_rouge_results["rouge-L"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.test_abs_rouge.reset()
        self.save_txt(self.args.val_save_file + '_reference', reference)
        self.save_txt(self.args.val_save_file + '_summary', summary)

    def test_step(self, batch, batch_idx):
        # batch
        # get summary
        batch['tokenized_contents'].pop('labels')
        summary_ids = self.model.generate(**batch['tokenized_contents'],
                                          num_beams=self.args.num_beams,
                                          max_length=self.args.max_output_len,
                                          #early_stopping=True,
                                          no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                                          )
        return [summary_ids, batch['refers']]

    def test_epoch_end(self, outputs):
        summary = []
        reference = []
        for item in outputs:
            try:
                summary_id = item[0]
                if self.args.tgt_lang != "zh_CN":
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]
                    self.test_abs_rouge.update(one_summary, item[1])
                else:
                    one_summary = [self.tokenizer.decode([i for i in g if i != -100], skip_special_tokens=True,
                                                         clean_up_tokenization_spaces=False) for g in summary_id]

                    self.test_abs_rouge.update([' '.join(self.tokenizer.tokenize(sum))[1:] for sum in one_summary],
                                               [' '.join(self.tokenizer.tokenize(sum))[1:] for sum in item[1]])
                summary += one_summary
                reference += item[1]
            except:
                print("某个生成出错啦")
        test_abs_rouge_results = self.test_abs_rouge.compute()
        self.log('test_R1', test_abs_rouge_results["rouge-1"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('test_R2', test_abs_rouge_results["rouge-2"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.log('test_RL', test_abs_rouge_results["rouge-L"]["f"], on_epoch=True, prog_bar=True,
                 sync_dist=True)
        self.test_abs_rouge.reset()
        self.save_txt(self.args.test_save_file, summary)
        print(calculate_rouge(summary, reference))


    def calrouge(self, summary, reference, rouge):
        rouge.add_batch(predictions=summary, references=reference)
        final_results = rouge.compute(rouge_types=["rouge1", "rouge2", "rougeL"])
        R1_F1 = final_results["rouge1"].mid.fmeasure * 100
        R2_F1 = final_results["rouge2"].mid.fmeasure * 100
        RL_F1 = final_results["rougeL"].mid.fmeasure * 100
        return R1_F1, R2_F1, RL_F1

    def save_txt(self, file_name, list_data):
        file = open(file_name, 'w',encoding='utf-8')
        list_data = [item+'\n' for item in list_data]
        file.writelines(list_data)
        file.close()

    def configure_optimizers(self):
        if self.args.ch2en_mode == 'dd_summary':
            self.args.learning_rate = self.args.learning_rate / 2
        if self.args.is_prompt:
            need_learning_para = []
            for name, param in self.model.named_parameters():
                if name != 'model.encoder.embed_tokens.learned_embedding':
                    param.requires_grad = False
                else:
                    need_learning_para.append(param)
            optimizer = torch.optim.Adam(need_learning_para, lr=self.args.learning_rate,)
        elif self.args.is_setAllturn is False and (self.args.is_peft_prompt or self.args.is_prefix or self.args.is_lora):
            need_learning_para = []
            all_para = [p for p in self.model.parameters()]
            for para in all_para:
                if para.requires_grad == True:
                    need_learning_para.append(para)
            optimizer = torch.optim.Adam(need_learning_para, lr=self.learning_rate)
        else:
            if self.args.is_setAllturn:
                for para in self.model.parameters():
                    para.requires_grad = True
            if self.args.loss_type == 46 or self.args.loss_type == 48 or self.args.loss_type == 49:
                optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.args.learning_rate},
                                              {'params': self.atw.parameters(), 'lr': self.args.learning_rate}],
                                             lr=self.learning_rate)
            else:
                optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1,
                                                    gamma=self.args.scheduler_lambda2, verbose=True)
        return [optimizer], [scheduler]


    def add_peft(self, args):
        self.args = args
        if args.is_prompt:
            self.model.model.encoder.embed_tokens = SoftEmbedding(self.model.model.encoder.embed_tokens,
                                                                  args.prompt_token_num)
        if args.is_peft_prompt:
            peft_config = PromptTuningConfig(
                peft_type="PROMPT_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=args.prompt_token_num,
                num_transformer_submodules=1,
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        if args.is_prefix:
            peft_config = PrefixTuningConfig(
                peft_type="PREFIX_TUNING", task_type="SEQ_2_SEQ_LM", num_virtual_tokens=args.prompt_token_num,
                num_transformer_submodules=1
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        if args.is_lora:
            peft_config = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=args.lora_r, lora_alpha=32, lora_dropout=0.05
            )
            self.model = get_peft_model(self.model, peft_config)
            self.model.print_trainable_parameters()
        else:
            raise ValueError('Wrong peft type!')

