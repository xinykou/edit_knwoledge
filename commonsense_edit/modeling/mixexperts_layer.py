#
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy

class MixtureSoup(torch.nn.Module):

    def __init__(self, expert, num_local_experts=1, using_random_experts=False):
        super(MixtureSoup, self).__init__()

        self.experts_module = torch.nn.ModuleList([copy.deepcopy(expert) for _ in range(num_local_experts)])
        self.num_local_experts = num_local_experts
        # 当使用 随机专家时，则不需要 学习一个 各个专家的权重
        self.expert_score_weight = torch.nn.Parameter(torch.zeros(self.num_local_experts), requires_grad=not using_random_experts)

    def get_expert_by_idx(self, idx):
        return self.experts_module[idx]

    def expert_soup_forward(self, input):
        output = F.linear(input,
                          self.parameter_dict["weight"],
                          self.parameter_dict["bias"])
        return output

    def expert_soup(self):
        weight = F.softmax(self.expert_score_weight, dim=-1)
        self.parameter_dict = {"weight": 0, "bias": 0}
        for idx in range(self.num_local_experts):
            single_expert = self.experts_module[idx]
            for s_name, s_param in single_expert.named_parameters():
                if "weight" in s_name:
                    p_name = "weight"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)
                else:
                    p_name = "bias"
                    self.parameter_dict[p_name] = self.parameter_dict[p_name] + (weight[idx] * s_param)


    def forward(self, hidden_states=None):
        expert_output = None
        if self.experts_module[0].training:  # 训练时，
            if self.expert_score_weight.requires_grad:
                self.expert_soup()  # 如果是给予 experts 不同的权重得到一个混合的矩阵W 和 Bias
                expert_output = self.expert_soup_forward(hidden_states)  # 通过 w 和 bias计算输出
            else:
                expert_idx = torch.randint(low=0, high=self.num_local_experts, size=(1,)).item()  # selected expert
                expert_output = self.get_expert_by_idx(expert_idx)(hidden_states)
        else:
            self.expert_soup()  # 预测时 使用的是 各个 experts的混合矩阵
            expert_output = self.expert_soup_forward(hidden_states)

        return expert_output



class MixExperts_Operator(nn.Module):

    def __init__(self, dim, r, num_expert=4, sharing_down=False, sharing_up=True, using_random_experts=True):
        super().__init__()

        if sharing_down:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), 1, using_random_experts)
        else:
            self.MoA_A = MixtureSoup(nn.Linear(dim, r), num_expert, using_random_experts)

        self.act = nn.GELU()

        if sharing_up:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), 1, using_random_experts)
        else:
            self.MoA_B = MixtureSoup(nn.Linear(r, dim), num_expert, using_random_experts)

    def forward(self, x, residual):
        result = self.MoA_A(x)
        if self.act is not None:
            result = self.act(result)
        result = self.MoA_B(result)
        return result + residual
