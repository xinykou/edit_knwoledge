import logging
import os
import shutil
import tempfile
import time
import json

import torch
from torch.utils.data import Dataset

from editors.utils import safe_backward
from tqdm import tqdm
from torch.utils.data import DataLoader
from editors.losses import kl_loc_loss, masked_log_probs
import pandas as pd
from dataset.dataset import DisentQADataset
from metric import query_model_mend
from metric import extract_model_answer_cols_from_new_format, compare_prediction_to_gold_per_row, get_total_score_means

LOG = logging.getLogger(__name__)


def run_eval(path, answer_type, train_stage=None):
    df = pd.read_csv(path)
    df = extract_model_answer_cols_from_new_format(df, answer_type, train_stage)  # 分割模型的输出结果
    df = compare_prediction_to_gold_per_row(df, answer_type, train_stage)   # 计算F1
    results_df = get_total_score_means(df, answer_type, train_stage)
    return results_df


class BaseTrainer:
    def __init__(self, model, config, train_set: Dataset, val_set=None, device=None):
        self.model = model  # 代表"MEND"或者ENN等
        self.config = config

        self.original_model = self.model.model_constructor()
        self.original_model.load_state_dict(self.model.pure_model.state_dict())
        self.original_model.to(device)

        self.model.to(device)
        self.device = device
        self.train_set = train_set
        self.val_set = val_set
        self.train_dataloader = DataLoader(self.train_set,
                                      batch_size=self.config.model.fine_tune.batch_size,
                                      shuffle=True,
                                      num_workers=4)

        self.OptimizerClass = getattr(torch.optim, config.model.fine_tune.opt)  # Adam 优化器
        LOG.info(f"Building optimizer {self.OptimizerClass} with lr {config.model.fine_tune.learning_rate}")
        self.global_iter = 0

    def run(self):
        for global_iter in range(0, self.config.model.fine_tune.n_epochs):
            for i, batch in tqdm(enumerate(self.train_dataloader), desc="Training"):
                labels = batch["labels"].to(self.device)
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                tokens = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels
                }
                self.train_step(tokens)  # todo: 运行训练
                self.global_iter += 1



class EditTrainer(BaseTrainer):
    def __init__(self, model, config, train_set: Dataset, val_set=None, device=None):
        super().__init__(model, config, train_set, val_set, device)
        # todo: 编辑器各个层的学习率 + mend 网络学习率 设置
        self.opt = self.OptimizerClass(self.model.outer_parameters(), lr=config.model.fine_tune.learning_rate)

    def edit_step(self, batch, loc, training: bool):
        self.model.train(training)  # 调整模型状态是---训练（是一个整体：包括原始模型，一个复制品 + 一个编辑器）
        self.original_model.train(training)  # 原始模型专题太调整为--训练

        new_loc = {k: v.to(self.device) for k, v in loc.items()}
        with torch.no_grad():
            base_logits = self.model.pure_model(**new_loc).logits  # todo: 编辑前在数据 “Xloc”原始模型:输出 batch * seq_len * voc_size
        # Do the edit
        # # "---执行编辑 训练---"， 梯度不更新的那个模型的参数更新了
        edited_model = self.model.editing(batch).pure_model

        with torch.set_grad_enabled(training):
            # 1. Editing loss
            post_edit = edited_model(**batch)  # 编辑后模型对于 “编辑样本相近的那些样本”的输出：batch * seq_len * voc_size
            l_edit = post_edit.loss  # 计算损失

            # 2. Locality loss, 模型更新前后计算 kl, 对于 Xloc数据尽量不变
            post_base_logits = edited_model(**new_loc).logits  # todo: 编辑后在数据 “Xloc”原始模型:输出 batch * seq_len * voc_size
            kl_mask = new_loc.get("decoder_attention_mask", new_loc["labels"])
            # 将 -100 转换为 0，其他值转换为 1
            converted_kl_mask = torch.where(kl_mask == -100, torch.tensor(0).to(self.device), torch.tensor(1).to(self.device))
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=converted_kl_mask)  # todo: 数据“Xloc”编辑前和编辑后计算“KL loss”

        l_total_edit = self.config.editor.cedit * l_edit + self.config.editor.cloc * l_loc  # loss = Le + Lloc
        # todo: 梯度回传
        if training:
            safe_backward(l_total_edit, self.model.outer_parameters(), self.config.model.fine_tune.accumulate_bs)   # 为了更新 mend的那些编辑器和 各个层的不同学习率

    def train_step(self, tokens):
        self.loc = self.train_set.generater_retention()  # 构建的关于 retention 样本
        self.edit_step(tokens, self.loc, training=True)  # todo: 真正编辑的步骤

        if self.global_iter > 0 and self.global_iter % self.config.model.fine_tune.accumulate_bs == 0:  # 累计一定批次才进行模型更新
            # todo: 更新 “编辑器” + "学习率" 两部分参数 的 参数
            self.opt.step()
            self.opt.zero_grad()


    def evaluate(self, tokenizer, train_path, train_config, inf_config, checkpoint_filename, train_source):
        # ------评估部分---------------------------------------------------------------------------------
        # todo: 如果是 把模型的结构更改了，训练后则直接评估，不在分成“训练” 和 “测试两部分”
        test_source = self.config.test_type
        test_path = self.config.experiment[test_source]

        train_dataset = DisentQADataset(self.config, tokenizer, train_path,
                                        source_max_token_len=train_config.source_max_token_len,
                                        target_max_token_len=train_config.target_max_token_len,
                                        type='test')
        val_path = train_path.replace('train.csv', 'test_retention.csv')
        val_dataset = DisentQADataset(self.config, tokenizer, val_path,
                                      source_max_token_len=train_config.source_max_token_len,
                                      target_max_token_len=train_config.target_max_token_len,
                                      type='test')

        test_dataset = DisentQADataset(self.config, tokenizer, test_path,
                                       source_max_token_len=inf_config.input_max_length,
                                       target_max_token_len=inf_config.output_max_length,
                                       type='test')

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=inf_config.batch_size,
                                      shuffle=False, num_workers=4,
                                      collate_fn=None)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=inf_config.batch_size,
                                    shuffle=False, num_workers=4,
                                    collate_fn=None)  # todo: 这里的样本是为了验证  ”不需要更改的样本的保持率“
        test_dataloader = DataLoader(test_dataset,
                                     batch_size=inf_config.batch_size,
                                     shuffle=False, num_workers=4,
                                     collate_fn=None)

        save_path_train_dataset = query_model_mend(self.model.pure_model, tokenizer, train_dataset,
                                              train_dataloader, inf_config, checkpoint_filename, type='train',
                                              device=self.device)
        # todo: 计算 训练集  编辑后 F1指标
        result_train = run_eval(save_path_train_dataset, answer_type=train_source)

        save_path_val_dataset = query_model_mend(self.model.pure_model, tokenizer, val_dataset, val_dataloader,
                                            inf_config, checkpoint_filename, type='val', device=self.device)
        # todo: 计算 验证集 也就是 预训练后需要保持不变的那个样本 的 F1指标
        result_val = run_eval(save_path_val_dataset, answer_type=train_source)

        save_path_test_dataset = query_model_mend(self.model.pure_model, tokenizer, test_dataset, test_dataloader,
                                             inf_config, checkpoint_filename, type='test', device=self.device)
        # todo: 计算 测试集 推理  F1指标
        result_test = run_eval(save_path_test_dataset, answer_type=test_source)

        print(f"--------Train dataset Metric:---------------")
        for k in result_train:
            print(f"{k}: {result_train[k]}")

        print(f"--------Retention dataset Metric:---------------")
        for k in result_val:
            print(f"{k}: {result_val[k]}")

        print(f"--------Test dataset Metric:---------------")
        for k in result_test:
            print(f"{k}: {result_test[k]}")
