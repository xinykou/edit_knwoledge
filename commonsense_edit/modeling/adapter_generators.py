import math

import torch
import torch.nn as nn

""""
超网络结构，用来生成 adapter 的 矩阵权重
"""
def hyperfanin_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


def hyperfanin_init_bias(linear_layer, hypernet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)


class SimpleGenerator(nn.Module):
    def __init__(self, config, input_dim, hidden_size, is_encoder=False):
        super().__init__()
        adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        self.input_dim = input_dim
        self.hidden_dim = config.hypernetwork_bottleneck    # 转化为超参数维度
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)  # batch * (1024 +10)  -> batch * 1024
        self.activation_fn = nn.ReLU()
        # output weights
        self.weight_up = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        self.weight_down = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, hidden_size)
        self.bias_down = nn.Linear(self.hidden_dim, adapter_dim)
        # init weights
        hyperfanin_init_weight(self.weight_up, self.hidden_dim, adapter_dim)
        hyperfanin_init_weight(self.weight_down, self.hidden_dim, hidden_size)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return (
            self.weight_up(x),
            self.weight_down(x),
            self.bias_up(x),
            self.bias_down(x),
        )

# todo: 只使用超网络 生成 decoder的 adapter参数
class ParameterGenerator(nn.Module):
    def __init__(self, config, hidden_size, is_encoder=False, only_using_last_layer=False):
        super().__init__()
        self.config = config
        self.only_using_last_layer = only_using_last_layer
        if only_using_last_layer:
            self.layer_embed = nn.Embedding(1, 10)
        else:  # decoder 所有层都添加 lora，
            self.layer_embed = nn.Embedding(config.num_hidden_layers, 10)  # 这个是层 可学习参数， 每个层编码后维度是 10
        self.decoder = SimpleGenerator(
            config, config.hidden_size + 10, hidden_size, is_encoder=is_encoder
        )

    def forward(self, hidden_inputs):
        layers = []
        # setup idxs we need
        layers_idxs = torch.arange(
            0,
            self.config.num_hidden_layers,
            dtype=torch.long,
            device=hidden_inputs.device,
        )
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)

        if self.only_using_last_layer:  # todo: decoder 只有最后一层 添加 lora
            n_layers = 1
        else:
            n_layers = self.config.num_hidden_layers

        for i in range(n_layers):
            layer_embed = self.layer_embed(layers_idxs[:, i])
            hidden_input = torch.cat([hidden_inputs, layer_embed], dim=1)
            layers.append(self.decoder(hidden_input))
        return layers


def linear_init_weight(linear_layer, hypernet_in, mainnet_in):
    bound = 1e-3 * math.sqrt(3 / (hypernet_in * mainnet_in))
    nn.init.uniform_(linear_layer.weight, -bound, bound)
    nn.init.constant_(linear_layer.bias, 0.0)





class SimpleGenerator_Bias(nn.Module):
    def __init__(self, config, input_dim, hidden_size, is_encoder=False):
        super().__init__()
        adapter_dim = (
            config.encoder_adapter_dim if is_encoder else config.decoder_adapter_dim
        )
        self.input_dim = input_dim
        self.hidden_dim = config.hypernetwork_bottleneck    # 转化为超参数维度
        self.linear1 = nn.Linear(self.input_dim, self.hidden_dim)  # batch * (1024 +10)  -> batch * 1024
        self.activation_fn = nn.ReLU()
        # output weights
        # self.weight_up = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        # self.weight_down = nn.Linear(self.hidden_dim, hidden_size * adapter_dim)
        self.bias_up = nn.Linear(self.hidden_dim, hidden_size)
        # self.bias_down = nn.Linear(self.hidden_dim, adapter_dim)
        # init weights
        # hyperfanin_init_weight(self.weight_up, self.hidden_dim, adapter_dim)
        # hyperfanin_init_weight(self.weight_down, self.hidden_dim, hidden_size)
        hyperfanin_init_bias(self.bias_up, self.hidden_dim)
        # hyperfanin_init_bias(self.bias_down, self.hidden_dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation_fn(x)
        return (
            None,
            None,
            None,
            self.bias_up(x)
        )

# todo: 只使用超网络 生成 decoder的 adapter参数
class ParameterGenerator_Bias(nn.Module):
    def __init__(self, config, hidden_size, is_encoder=False, only_using_last_layer=False):
        super().__init__()
        self.config = config
        self.only_using_last_layer = only_using_last_layer
        if only_using_last_layer:
            self.layer_embed = nn.Embedding(1, 10)
        else:  # decoder 所有层都添加 lora，
            self.layer_embed = nn.Embedding(config.num_hidden_layers, 10)  # 这个是层 可学习参数， 每个层编码后维度是 10
        self.decoder = SimpleGenerator_Bias(
            config, config.hidden_size + 10, hidden_size, is_encoder=is_encoder
        )

    def forward(self, hidden_inputs):
        layers = []
        # setup idxs we need
        layers_idxs = torch.arange(
            0,
            self.config.num_hidden_layers,
            dtype=torch.long,
            device=hidden_inputs.device,
        )
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)

        if self.only_using_last_layer:  # todo: decoder 只有最后一层 添加 lora
            n_layers = 1
        else:
            n_layers = self.config.num_hidden_layers

        for i in range(n_layers):
            layer_embed = self.layer_embed(layers_idxs[:, i])
            hidden_input = torch.cat([hidden_inputs, layer_embed], dim=1)
            layers.append(self.decoder(hidden_input))
        return layers




# # todo：使用 超网络 生成 adapter参数，同时 experts生成 scale + shift 用来加强调整 adapter参数
class ParameterGenerator_with_expert_by_scale_shift(nn.Module):
    def __init__(self, config, hidden_size, is_encoder=False):
        super().__init__()
        self.config = config
        self.layer_embed = nn.Embedding(config.num_hidden_layers, 10)  # 这个是层 可学习参数， 每个层编码后维度是 10
        self.decoder_adapter = SimpleGenerator(config, config.hidden_size + 10, hidden_size, is_encoder=is_encoder)
        # dim = hidden_size * config.decoder_adapter_dim  # 1024 * 64
        # self.adapter_dim = config.decoder_adapter_dim
        # self.to_scale_up = nn.Linear(hidden_size + 10, dim)
        # self.to_shift_up = nn.Linear(hidden_size + 10, dim)
        # self.to_scale_down = nn.Linear(hidden_size + 10, dim)
        # self.to_shift_down = nn.Linear(hidden_size + 10, dim)
        #
        # linear_init_weight(self.to_scale_down, hidden_size + 10, dim)
        # linear_init_weight(self.to_shift_down, hidden_size + 10, dim)
        # linear_init_weight(self.to_scale_up, hidden_size + 10, dim)
        # linear_init_weight(self.to_shift_up, hidden_size + 10, dim)
        self.decoder_adapter_from_experts = SimpleGenerator(config, config.hidden_size + 10, hidden_size, is_encoder=is_encoder)


    def forward(self, hidden_inputs, other_context_knowledge=None, aug_attention_mask=None):
        layers = []
        layers_other = []

        layers_idxs = torch.arange(
            0,
            self.config.num_hidden_layers,
            dtype=torch.long,
            device=hidden_inputs.device,
        )
        layers_idxs = layers_idxs.repeat(hidden_inputs.size(0), 1)
        for i in range(self.config.num_hidden_layers):
            layer_embed = self.layer_embed(layers_idxs[:, i])
            hidden_input = torch.cat([hidden_inputs, layer_embed], dim=1)  # batch * (1024+10)
            org_adapter = self.decoder_adapter(hidden_input)  # batch* (1024 * 64), batch* (1024 * 64), batch * 1024, batch * 1024

            experts_inputs_with_layer_embed = torch.cat([other_context_knowledge, layer_embed], dim=1)
            # scale_down = self.to_scale_down(experts_inputs_with_layer_embed)  # batch* (1024 * 64)
            # shift_down = self.to_shift_down(experts_inputs_with_layer_embed)  # batch* (1024 * 64)
            #
            # scale_up = self.to_scale_up(experts_inputs_with_layer_embed)   # batch* (1024 * 64)
            # shift_up = self.to_shift_up(experts_inputs_with_layer_embed)   # batch* (1024 * 64)
            other_adapter = self.decoder_adapter_from_experts(experts_inputs_with_layer_embed)

            layers.append(org_adapter)
            layers_other.append(other_adapter)

        return layers, layers_other

