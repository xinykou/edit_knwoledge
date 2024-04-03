import torch
import pytorch_lightning as pl
import transformers
from transformers import get_linear_schedule_with_warmup, T5ForConditionalGeneration, T5Tokenizer
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
import time

torch.multiprocessing.set_sharing_strategy('file_system')
class prototype_module(pl.LightningModule):
    def __init__(self, args=None, data_num=None, device=None):
        super(prototype_module, self).__init__()

        self.devices = device
        print(f"Loading model class from cache dir {args.model_cache}")
        self.model = T5ForConditionalGeneration.from_pretrained(args.model_cache).cuda()
        for p in self.model.parameters():  # todo: 模型的梯度全部冻结
            p.requires_grad = False

        self.tokenizer = T5Tokenizer.from_pretrained(args.model_cache)

        self.config = args
        self.lr = args.learning_rate
        self.warmup_step = args.n_epoch * (data_num // args.batch_size) // 100
        self.all_step = args.n_epoch * (data_num // args.batch_size)

        w = torch.empty((data_num, args.proto_dim))  # todo: 原型向量
        nn.init.xavier_uniform_(w)
        self.proto = nn.Parameter(w, requires_grad=True)  # todo: 为什么不需要梯度 ？？？？
        initial_r = torch.full((data_num,), 0.5)
        self.head = nn.Linear(1024, args.proto_dim, bias=False)  # 1024 代表 编码器的隐含层的维度
        self.proto_r = nn.Parameter(initial_r, requires_grad=True)
        param_groups = [
            {'params': self.head.parameters()},
            {'params': self.proto_r}
        ]
        self.optimizer = torch.optim.Adam(param_groups, lr=self.lr)

        self.num_classes = data_num   # todo: 所有的类别



    def sim(self, x, y, r=0, model_logits=0, model_logits_weight=1):
        x = F.normalize(x, p=2, dim=-1)   #  (batch* num) *1* dim
        num = x.size(0)
        n_class = y.size(0)
        x = x.expand(-1, n_class, -1)
        y = y.unsqueeze(0).expand(num, -1, -1)
        dist = -torch.norm((x - y), dim=-1) + r
        return dist

    def loss_func(self, x, labels):
        sim_mat = torch.exp(self.sim(x, self.proto, self.proto_r))  # (batch* num) * dim
        class_mask = F.one_hot(labels, self.num_classes)
        pos_score = torch.sum(sim_mat * class_mask, -1)
        loss = -torch.mean(torch.log(pos_score / sim_mat.sum(-1)))
        return loss

    def train_proto(self, embedding_dataloader):
        start_time = time.time()
        loss = 0.

        embeds = [[] for _ in range(self.num_classes)]
        labels = [[] for _ in range(self.num_classes)]
        for batch in tqdm(embedding_dataloader):
            input_ids = batch['input_ids'].cuda()
            input_attention_mask = batch['attention_mask'].cuda()
            torch.cuda.empty_cache()
            with torch.no_grad():
                encoder_outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=input_attention_mask)
            # 取最后一个 token 编码
            list_num = [i for i in range(len(encoder_outputs[0]))]
            for j in list_num:
                nonpad_positions = torch.nonzero(input_ids[j] != self.tokenizer.pad_token_id)
                # 找到最后一个非pad元素的编码
                last_nonpad_element = nonpad_positions[-1, :]
                last_token_encoding = encoder_outputs[0][j, last_nonpad_element, :]
                label = batch['labels'][j]  # 因为只有一个样本，所以标签只有一个
                labels[label].append(label)
                embeds[label].append(last_token_encoding)

        self.embeds = list(map(torch.stack, embeds))  # 列表中每个元素是 num * 1 * 1024, 其中num表示每个类下的样本数量
        self.labels = torch.cat(list(map(torch.stack, labels))).to(self.devices)  # tensor (0,1, 2....
        print()

        for epoch in range(self.config.n_epoch):
            for index in range(len(self.embeds)):
                x = self.head(self.embeds[index])
            # (batch * num) * dim
            # x = self.head(torch.cat(self.embeds))
                self.optimizer.zero_grad()
                loss = self.loss_func(x, self.labels[index:index+1])
                loss.backward()
                self.optimizer.step()
                print(f"train_loss:{loss}")

        print("Total epoch: {}. DecT loss: {}".format(self.config.n_epoch, loss))
        end_time = time.time()
        print("Training time: {}".format(end_time - start_time))

    def test(self, dataloader):

        self.model.eval()

        embeds = [[] for _ in range(self.num_classes)]
        labels = [[] for _ in range(self.num_classes)]
        for batch in tqdm(dataloader):
            input_ids = batch['input_ids'].cuda()
            input_attention_mask = batch['attention_mask'].cuda()
            torch.cuda.empty_cache()
            with torch.no_grad():
                encoder_outputs = self.model.get_encoder()(input_ids=input_ids, attention_mask=input_attention_mask)
            # 取最后一个 token 编码
            list_num = [i for i in range(len(encoder_outputs[0]))]
            for j in list_num:
                nonpad_positions = torch.nonzero(input_ids[j] != self.tokenizer.pad_token_id)
                # 找到最后一个非pad元素的编码
                last_nonpad_element = nonpad_positions[-1, :]
                last_token_encoding = encoder_outputs[0][j, last_nonpad_element, :]
                label = batch['labels'][j]  # 因为只有一个样本，所以标签只有一个
                labels[label].append(label)
                embeds[label].append(last_token_encoding)

        embeds = list(map(torch.stack, embeds))  # 列表中每个元素是 num * 1 * 1024, 其中num表示每个类下的样本数量
        labels = torch.cat(list(map(torch.stack, labels))).to(self.devices)  # tensor (0,1, 2....
        preds = []

        x = self.head(torch.cat(embeds))  # (batch * num) * dim
        dist = self.sim(x, self.proto, self.proto_r)
        pred_idx = torch.argmax(dist, dim=-1)
        max_value = dist[torch.arange(dist.shape[0]), pred_idx]
        # 查找小于阈值的位置
        indices = torch.nonzero(max_value < self.config.threshold).squeeze()
        pred_idx[indices] = -1
        preds.extend(pred_idx.cpu().tolist())
        labels = labels.cpu().tolist()

        acc = self.accuracy(labels, pred_idx)
        print(f"准确率：{acc}")


    def accuracy(self, labels, predictions):
        """
        计算标签和预测列表重合的准确率

        Args:
            labels (list): 包含所有真实标签的列表
            predictions (list): 包含所有预测标签的列表

        Returns:
            float: 准确率
        """
        assert len(labels) == len(predictions), "标签和预测列表长度必须相同"

        correct = 0
        for i in range(len(labels)):
            if labels[i] == predictions[i]:
                correct += 1

        accuracy = correct / len(labels)
        return accuracy


