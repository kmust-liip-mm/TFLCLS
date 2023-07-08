# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
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

import logging
from rouge import Rouge
from torchmetrics import Metric
import torch
# import nltk

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target, acc_len=0):
        assert len(preds) == len(target)
        correct = 0.0

        if acc_len > 0:
            preds = preds[:acc_len]
            target = target[:acc_len]
            for pred in preds:
                if pred in target:
                    correct += 1
            self.correct += correct / len(target)
        else:
            correct += sum([1 for i in range(len(preds)) if preds[i] == target[i]])
            self.correct += correct / len(target)

        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class ROUGEScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # self.add_state("preds", default=[], dist_reduce_fx="cat")
        # self.add_state("refers", default=[], dist_reduce_fx="cat")
        self.add_state("r1_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r1_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r1_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("r2_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r2_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r2_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rl_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rl_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rl_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.rouge = Rouge()

    def update(self, preds, refers):
        assert len(preds) == len(refers)
        rouge_results = self.rouge.get_scores(preds, refers)
        for rouge_result in rouge_results:
            self.r1_p += rouge_result['rouge-1']['p']
            self.r1_r += rouge_result['rouge-1']['r']
            self.r1_f += rouge_result['rouge-1']['f']

            self.r2_p += rouge_result['rouge-2']['p']
            self.r2_r += rouge_result['rouge-2']['r']
            self.r2_f += rouge_result['rouge-2']['f']

            self.rl_p += rouge_result['rouge-l']['p']
            self.rl_r += rouge_result['rouge-l']['r']
            self.rl_f += rouge_result['rouge-l']['f']

        self.total += len(preds)
        return rouge_results

    def compute(self):
        # return rouge_results
        return {
            'total': self.total.item(),
            'rouge-1': {
                'f': (self.r1_f / self.total).item(),
                'p': (self.r1_p / self.total).item(),
                'r': (self.r1_r / self.total).item(),
            },
            'rouge-2': {
                'f': (self.r2_f / self.total).item(),
                'p': (self.r2_p / self.total).item(),
                'r': (self.r2_r / self.total).item(),
            },
            'rouge-L': {
                'f': (self.rl_f / self.total).item(),
                'p': (self.rl_p / self.total).item(),
                'r': (self.rl_r / self.total).item(),
            }
        }