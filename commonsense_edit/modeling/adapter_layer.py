import torch
from torch import nn
import math


class AdapterLayer(nn.Module):
    def __init__(self, config, is_encoder=False):
        super().__init__()
        self.adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        hidden_size = config.hidden_size
        self.config = config
        self.input_dim = config.hidden_size
        self.output_dim = config.hidden_size
        # insertion weights
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual = nn.Linear(hidden_size, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, hidden_size)
        # 如果冻结 down
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)

        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)

    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, uw, dw, ub, db):
        # try:
        #     fz_down = self.config.freeze_decoder_adapters_down
        # except:
        #     fz_down = False
        # if fz_down:
        self.adapter_down_weight = dw.view(bsz, self.input_dim, self.adapter_dim)
        self.adapter_down_bias = db.view(bsz, self.adapter_dim)
        self.adapter_up_weight = uw.view(bsz, self.adapter_dim, self.output_dim)
        self.adapter_up_bias = ub.view(bsz, self.output_dim)

    def forward(self, x):
        if self.adapter_down_weight is not None:
            # if self.config.freeze_decoder_adapters_down:  # todo：如果冻结down lora, 则 down lora不在是超网络生成，而是 直接初始化的
            #     x = self.adapter_down_manual(x)
            # else:
            x = (x @ self.adapter_down_weight) + self.adapter_down_bias.unsqueeze(1)  # 代表矩阵乘法
            x = self.hidden_act(x)
            x = (x @ self.adapter_up_weight) + self.adapter_up_bias.unsqueeze(1)
        else:
            x = self.adapter_down_manual(x)
            x = self.hidden_act(x)
            x = self.adapter_up_manual(x)
        return x  # no residual connection - we let the user of this layer decide that



class AdapterLayer_Bias(nn.Module):
    def __init__(self, config, is_encoder=False):
        super().__init__()
        self.adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        hidden_size = config.hidden_size
        self.config = config
        self.input_dim = config.hidden_size
        self.output_dim = config.hidden_size
        # insertion weights
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual = nn.Linear(hidden_size, self.adapter_dim)
        self.adapter_up_manual = nn.Linear(self.adapter_dim, hidden_size)
        # 如果冻结 down
        nn.init.xavier_uniform_(self.adapter_up_manual.weight, gain=1e-4)
        nn.init.xavier_uniform_(self.adapter_down_manual.weight, gain=1e-4)

        nn.init.constant_(self.adapter_up_manual.bias, 0.0)
        nn.init.constant_(self.adapter_down_manual.bias, 0.0)

        # todo: 额外添加的 bias
        self.adapter_alpha = nn.Linear(hidden_size, 1)  # 每個layer的xi都不一樣
        nn.init.xavier_uniform_(self.adapter_alpha.weight, gain=1e-4)
        nn.init.constant_(self.adapter_alpha.bias, 0.0)


    def clear_adapter(self):
        self.adapter_down_weight = None
        self.adapter_down_bias = None
        self.adapter_up_weight = None
        self.adapter_up_bias = None

    def apply_adapter_params(self, bsz, uw=None, dw=None, ub=None, db=None):
        # self.adapter_down_weight = dw.view(bsz, self.input_dim, self.adapter_dim)
        # self.adapter_down_bias = db.view(bsz, self.adapter_dim)
        # self.adapter_up_weight = uw.view(bsz, self.adapter_dim, self.output_dim)
        # self.adapter_up_bias = ub.view(bsz, self.output_dim)
        self.adapter_up_bias = db.view(bsz, self.output_dim)

    def forward(self, x):  # batch * seq_len * dim
        a = x.size(0)
        b = x.size(1)
        c = x.size(2)  # dim
        y = self.adapter_alpha(x)  # batch * seq_len * 1
        y_expanded = y.expand(a, b, c)
        bias_expanded = self.adapter_up_bias.unsqueeze(1).expand(a, b, c)  # self.adapter_up_bias尺寸是 batch * dim
        hidden_states = y_expanded * bias_expanded
        return hidden_states  # no residual connection - we let the user of this layer decide that

class TaskSpecificAdapterLayer(nn.Module):
    def __init__(self, config, task_list, is_encoder=False):
        super().__init__()
        self.adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        hidden_size = config.hidden_size
        task_list = config.tasks
        self.input_dim = hidden_size
        self.output_dim = hidden_size
        self.hidden_act = nn.ReLU()
        # learnt adapter + inits for it
        self.adapter_down_manual_weight = nn.Parameter(
            torch.randn(len(task_list), hidden_size, self.adapter_dim)
        )
        self.adapter_down_manual_bias = nn.Parameter(
            torch.randn(len(task_list), 1, self.adapter_dim)
        )
        self.adapter_up_manual_weight = nn.Parameter(
            torch.randn(len(task_list), self.adapter_dim, hidden_size)
        )
        self.adapter_up_manual_bias = nn.Parameter(
            torch.randn(len(task_list), 1, hidden_size)
        )

        nn.init.xavier_uniform_(self.adapter_down_manual_weight, gain=1e-4)
        nn.init.constant_(self.adapter_down_manual_bias, 0.0)
        nn.init.xavier_uniform_(self.adapter_up_manual_weight, gain=1e-4)
        nn.init.constant_(self.adapter_up_manual_bias, 0.0)
        # hacky method for setting task specific adapters
        self.adapter_down_weight_holder = None
        self.adapter_down_bias_holder = None
        self.adapter_up_weight_holder = None
        self.adapter_up_bias_holder = None

    def clear_adapter(self):
        self.adapter_down_weight_holder = None
        self.adapter_down_bias_holder = None
        self.adapter_up_weight_holder = None
        self.adapter_up_bias_holder = None

    def set_indices(self, indices):
        self.adapter_down_weight_holder = self.adapter_down_manual_weight[indices]
        self.adapter_down_bias_holder = self.adapter_down_manual_bias[indices]
        self.adapter_up_weight_holder = self.adapter_up_manual_weight[indices]
        self.adapter_up_bias_holder = self.adapter_up_manual_bias[indices]

    def forward(self, x):
        x = (
            torch.bmm(x, self.adapter_down_weight_holder)
            + self.adapter_down_bias_holder
        )
        x = self.hidden_act(x)
        x = torch.bmm(x, self.adapter_up_weight_holder) + self.adapter_up_bias_holder
        return x



class PrefixLayer(nn.Module):
    def __init__(self, config, is_encoder=False):
        super().__init__()

        self.generator_type = config.generator_type
        self.num_prompt_tokens = config.num_prompt_tokens

        self.proj_down = nn.Linear(config.hidden_size, config.proj_down_size)
        self.intermediate_act_fn = nn.ReLU()
        if self.generator_type == 'MPPG':
            self.adaptive_pooling = nn.AdaptiveMaxPool1d(self.num_prompt_tokens)
        elif self.generator_type == 'APPG':
            self.adaptive_pooling = nn.AdaptiveAvgPool1d(self.num_prompt_tokens)
        self.proj_up = nn.Linear(config.proj_down_size, config.hidden_size)
        self.dropout = nn.Dropout(0.1)

    def clear_prefix(self):
        self.feature = None

    def apply_prefix_params(self, feature=None):
        self.feature = feature

    def forward(self):
        assert self.feature != None
        hidden_states = self.proj_down(self.feature)

        batch_prompts = []
        for i in range(hidden_states.size(0)):
            hidden_state = hidden_states[i].unsqueeze(0)
            hidden_state = hidden_state.transpose(1, 2)  # B x D x L
            hidden_state = (self.adaptive_pooling(hidden_state)).transpose(1, 2)  # B x num_prompt_tokens x D
            batch_prompts.append(hidden_state)

        hidden_states = torch.cat(batch_prompts, dim=0)

        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.proj_up(hidden_states)
        # hidden_states = self.dropout(hidden_states)

        return hidden_states