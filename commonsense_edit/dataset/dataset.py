import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# todo: 对于contextual 和 question 同时送入网络的 结构有效
class DisentQADataset(Dataset):
    def __init__(self, config, tokenizer, path, source_max_token_len=396, target_max_token_len=32, type=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data = pd.read_csv(path)
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.type = type  # 'train' or 'test'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        question_and_context = data_row['input']
        source_encoding = self.tokenizer(question_and_context, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        if self.type != 'test':
            answer_text = data_row['output']
            target_encoding = self.tokenizer(answer_text, max_length=self.target_max_token_len, padding="max_length",
                                                 truncation=True, return_attention_mask=True, add_special_tokens=True,
                                                 return_tensors="pt")
            labels = target_encoding["input_ids"].flatten()  # "answer_text"
            labels[labels == 0] = -100

        else:
            pass

        input_ids = source_encoding["input_ids"].flatten()  # question_and_context
        attention_mask = source_encoding["attention_mask"].flatten()  # question_and_context


        if self.type != 'test':
            return {"question_and_context": question_and_context, "answer_text": answer_text, "input_ids": input_ids,
                    "attention_mask": attention_mask, "labels": labels}
        else:
            return {"question_and_context": question_and_context, "input_ids": input_ids,
                    "attention_mask": attention_mask}


# todo: 对于contextual 和 question 分开编码，然后融合的结构有效
class Contextual_Question_Separated_Dataset(Dataset):
    def __init__(self, config, tokenizer, path, source_max_token_len=396, target_max_token_len=32, context_max_token_len=30, type='train'):
        self.tokenizer = tokenizer
        self.config = config
        self.data = pd.read_csv(path)
        self.source_max_token_len = source_max_token_len
        self.context_max_token_len = context_max_token_len
        self.target_max_token_len = target_max_token_len
        self.type = type  # 'train' or 'test'

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.iloc[index]

    def shuffle_unique_order(self, lst):
        new_list = lst.copy()
        while any(new_list[i] == lst[i] for i in range(len(lst))):
            random.shuffle(new_list)
        return new_list

    def collate_fn(self, batch):
        # todo: 默认设定 证据的数量是 ”2“
        try:
            context_num = self.config.num_aug_sources
        except:
            context_num = self.config.editor.num_aug_sources
        questions_list = []
        labels_list = []
        context_input_ids_list = []
        context_attention_mask_list = []
        try:
            random_evidence = self.config.random_evidence
        except:
            random_evidence = False
        try:
            noisy_evidence = self.config.noisy_evidence
        except:
            noisy_evidence = False

        try:
            random_add_noisy_evidence = self.config.random_add_noisy_evidence
        except:
            random_add_noisy_evidence = False

        batch_index = [j for j in range(len(batch))]
        noisy_li = self.shuffle_unique_order(batch_index)
        for i in range(len(batch)):
            question = batch[i]['question'].split('\n')[0]  # 我们这里只用第一个问题
            # if self.config.editor._name == "lora_hyperdecoders_postfusion_mixexperts":
            #     question = "question: " + batch[i]['question'].split('\n')[0]
            """判断是否 context的选择：1. 当 close book 类型 输入时 context是空； 2. 当 random 一个context从两个中， 
                                    3. 从其他样本采样一个context, 是noisy, 4. 当两个context都选择时， 
            """
            if batch[i]['contextual_answer'] == 'unanswerable':  #
                # contexts = [question for _ in range(context_num)]
                contexts = ["No information." for _ in range(context_num)]
            elif random_evidence and self.type == 'train':
                context_org = batch[i]['context'].split('\n')  # 一个列表中包含两个 证据
                li = [i for i in range(len(context_org))]
                contexts_index = random.choices(li, k=1)
                contexts = context_org[contexts_index[0]]
                contexts = ["Supporting knowledge is as follows." + contexts]
            elif noisy_evidence and self.type == 'train':
                inds = noisy_li[i]
                context_org = batch[inds]['context'].split('\n')
                contexts = ["Supporting knowledge is as follows." + context_org[0]]
            elif random_add_noisy_evidence and self.type == 'train':
                context_org = batch[i]['context'].split('\n')  # 一个列表中包含两个 证据
                li = [i for i in range(len(context_org))]
                contexts_index = random.choices(li, k=1)
                random_contexts = context_org[contexts_index[0]]

                inds = noisy_li[i]
                context_org = batch[inds]['context'].split('\n')
                noisy_contexts = context_org[0]

                contexts = ["Supporting knowledge is as follows." + random_contexts + \
                            "Supporting knowledge is as follows." + noisy_contexts]
            else:
                context_org = batch[i]['context'].split('\n')  # 一个列表中包含两个 证据
                contexts = ["Supporting knowledge is as follows." + ' ' + org_txt for org_txt in context_org]
                if len(contexts) == 1:
                    contexts = [contexts[0] for _ in range(context_num)]

            context_encoding = self.tokenizer(contexts, max_length=self.context_max_token_len,
                                              padding="max_length",
                                              truncation=True, return_attention_mask=True, add_special_tokens=True,
                                              return_tensors="pt")
            context_input_ids_list.append(context_encoding["input_ids"])
            context_attention_mask_list.append(context_encoding["attention_mask"])

            questions_list.append(question)

            if self.type != 'test':
                answer_text = batch[i]['output']
                labels_list.append(answer_text)

        source_encoding = self.tokenizer(questions_list, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")


        if self.type != 'test':
            target_encoding = self.tokenizer(labels_list, max_length=self.target_max_token_len, padding="max_length",
                                             truncation=True, return_attention_mask=True, add_special_tokens=True,
                                             return_tensors="pt")
            labels = target_encoding["input_ids"]
            labels_mask = target_encoding["attention_mask"]
            labels[labels == 0] = -100

        print()

        context_input_ids_all = torch.stack(context_input_ids_list)  # batch * num * seq_len,其中 num代表一个样本拥有的证据数量
        context_attention_mask_all = torch.stack(context_attention_mask_list)

        if self.type != 'test':
            return {"questions": questions_list,
                    "answer_text": labels_list,
                    "input_ids": source_encoding.input_ids,
                    "attention_mask": source_encoding.attention_mask,
                    "context_input_ids": context_input_ids_all,
                    "context_attention_mask": context_attention_mask_all,
                    "labels": labels,
                    "labels_mask": labels_mask}
        else:
            return {"input_ids": source_encoding.input_ids,
                    "attention_mask": source_encoding.attention_mask,
                    "context_input_ids": context_input_ids_all,
                    "context_attention_mask": context_attention_mask_all,
                    }


# todo: "Mend" 训练时，单独 一个 数据加载器
class Mend_Dataset(Dataset):
    def __init__(self, config, tokenizer, path, retention_path=None, source_max_token_len=396, target_max_token_len=32, type=None):
        self.tokenizer = tokenizer
        self.config = config
        self.data = pd.read_csv(path)
        self.data_retention = pd.read_csv(retention_path)
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len
        self.type = type  # 'train' or 'test'

        self.n_retention = config.model.fine_tune.retention_batch_size
        self.n = len(self.data_retention)
        self.rng = np.random.default_rng(seed=0)
        self.last_idx = [0 for _ in range(self.n_retention)]  # 初始化

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        question_and_context = data_row['input']
        source_encoding = self.tokenizer(question_and_context, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        if self.type != 'test':
            answer_text = data_row['output']
            target_encoding = self.tokenizer(answer_text, max_length=self.target_max_token_len, padding="max_length",
                                                 truncation=True, return_attention_mask=True, add_special_tokens=True,
                                                 return_tensors="pt")
            labels = target_encoding["input_ids"].flatten()  # "answer_text"
            labels[labels == 0] = -100

        else:
            pass

        input_ids = source_encoding["input_ids"].flatten()  # question_and_context
        attention_mask = source_encoding["attention_mask"].flatten()  # question_and_context

        if self.type != 'test':
            return {"question_and_context": question_and_context, "answer_text": answer_text, "input_ids": input_ids,
                    "attention_mask": attention_mask, "labels": labels}
        else:
            return {"question_and_context": question_and_context, "input_ids": input_ids,
                    "attention_mask": attention_mask}

    # 采样 保持的样本
    def sampler_retention(self):
        loc_idxs = self.rng.choice(self.n, self.n_retention)
        while len(np.intersect1d(self.last_idx, loc_idxs)) > 0:  # 获取编辑样本和 非 编辑样本重叠率
            loc_idxs = self.rng.choice(self.n, self.n_retention)  # 如果重叠，则需要新采样非编辑样本
        self.last_idx = loc_idxs
        return loc_idxs.tolist()

    def generater_retention(self):
        loc = {}
        loc_idx = self.sampler_retention()  # 采样用于该批次的 retention样本
        batch_retention = [self.data_retention.iloc[idx] for idx in loc_idx]
        questions = [b['input'] for b in batch_retention]
        answers = [b['output'] for b in batch_retention]
        source_encoding = self.tokenizer(questions, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        target_encoding = self.tokenizer(answers, max_length=self.target_max_token_len, padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        labels = target_encoding["input_ids"]   # "answer_text"
        labels[labels == 0] = -100
        input_ids = source_encoding["input_ids"]  # question_and_context
        attention_mask = source_encoding["attention_mask"]  # question_and_context
        loc["input_ids"] = input_ids
        loc["attention_mask"] = attention_mask
        loc["labels"] = labels
        return loc

class Prototype_Dataset(Dataset):
    def __init__(self, tokenizer, path, source_max_token_len=396, target_max_token_len=32, data_type=None):
        self.tokenizer = tokenizer
        org_data = pd.read_csv(path)
        if data_type == 'train':
            self.data = org_data[:2000]   # todo: 只取前 2000个数据用来训练
        elif data_type == 'test':
            self.data = org_data
        self.source_max_token_len = source_max_token_len
        self.target_max_token_len = target_max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index]
        question = data_row['input']
        source_encoding = self.tokenizer(question, max_length=self.source_max_token_len,
                                         padding="max_length",
                                         truncation=True, return_attention_mask=True, add_special_tokens=True,
                                         return_tensors="pt")
        input_ids = source_encoding["input_ids"].flatten()  # question_and_context
        attention_mask = source_encoding["attention_mask"].flatten()  # question_and_context

        need_or_not_edit = data_row['source_type']

        label = data_row['label']-1
        target = torch.tensor(label)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": target}

