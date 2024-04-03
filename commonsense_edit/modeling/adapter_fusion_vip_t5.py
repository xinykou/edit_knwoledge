# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
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

# modified modeling_t5 code.
import copy
import warnings
from einops import rearrange, repeat
from einops_exts import rearrange_many, repeat_many
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import inspect
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.utils import ModelOutput
from transformers.models.t5.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5Stack,
    T5ForConditionalGeneration,
    __HEAD_MASK_WARNING_MSG,
)

from .adapter_generators import ParameterGenerator, ParameterGenerator_with_expert_by_scale_shift
from .adapter_layer import AdapterLayer, TaskSpecificAdapterLayer
from .xattn_layer import GatedCrossAttentionBlock, PerceiverResampler, CrossAttentionBlock
from .xattn_layer import freeze_all_layers_, unfreeze_all_layers_
from .xattn_layer import Classify_for_adapter_Weight
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from transformers import T5Tokenizer, T5EncoderModel
from torch.distributions.multinomial import Multinomial



class T5WithAdapterConfig(T5Config):
    def __init__(
        self,
        encoder_adapter_dim=64,
        decoder_adapter_dim=64,
        hypernetwork_bottleneck=128,
        encoder_adapter="task",
        decoder_adapter="task",
        tasks=[],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.encoder_adapter_dim = encoder_adapter_dim
        self.decoder_adapter_dim = decoder_adapter_dim
        self.hypernetwork_bottleneck = hypernetwork_bottleneck
        self.encoder_adapter = encoder_adapter
        self.decoder_adapter = decoder_adapter
        self.tasks = tasks


class T5LayerFFWithAdapter(T5LayerFF):
    def __init__(self, config, is_encoder=False):
        super().__init__(config)
        self.config = config
        if (is_encoder and config.encoder_adapter == "manual_specific") or (
            not is_encoder and config.decoder_adapter == "manual_specific"
        ):
            self.adapter_layer = TaskSpecificAdapterLayer(config, is_encoder=is_encoder)
        else:
            self.adapter_layer = AdapterLayer(config, is_encoder=is_encoder)

    def forward(self, hidden_states):
        normed_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(normed_states)
        adapter_input = (
            normed_states if self.config.adapter_norm_input else hidden_states
        )
        hidden_states = (
            hidden_states
            + self.dropout(forwarded_states)
            + self.adapter_layer(adapter_input)
        )
        return hidden_states


class T5BlockWithAdapter(T5Block):
    def __init__(self, config, has_relative_attention_bias=False, is_encoder=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        self.layer[-1] = T5LayerFFWithAdapter(config, is_encoder=is_encoder)


def mean_pooling(hidden_state, attention_mask):
    input_masked = hidden_state * attention_mask.unsqueeze(-1)
    return input_masked.sum(1) / attention_mask.sum(1).unsqueeze(-1)

# encoder 或者 decoder
class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        blockClass = T5Block
        if (self.is_decoder and self.config.decoder_adapter != "none") or (
            (not self.is_decoder) and self.config.encoder_adapter != "none"
        ):
            blockClass = T5BlockWithAdapter   # todo: 执行这里，相当于在ffn层 部分加了 adapter
            kwargs = {"is_encoder": not self.is_decoder}
        else:
            kwargs = {}
        self.block = torch.nn.ModuleList(   # 每一层 拼接在一起，形成 编码器 或者解码器
            [
                blockClass(config, has_relative_attention_bias=bool(i == 0), **kwargs)
                for i in range(config.num_layers)
            ]
        )
        if (self.is_decoder and self.config.decoder_adapter == "generated") or (
            (not self.is_decoder) and self.config.encoder_adapter == "generated"
        ):

            # todo: 这里是用来 使用超网络生成 解码器需要的参数
            self.param_gen = ParameterGenerator(config, config.hidden_size, is_encoder=not self.is_decoder)
            if self.config.process_encoder_output:
                self.mlp = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.ReLU(),
                    nn.Linear(config.d_model, config.d_model),
                )
                self.mlp_experts = nn.Sequential(
                    nn.Linear(config.d_model, config.d_model),
                    nn.ReLU(),
                    nn.Linear(config.d_model, config.d_model),
                )

            else:
                # no-op to make the forward function less of an if-maze
                self.mlp = lambda x: x
        elif (self.is_decoder and self.config.decoder_adapter == "task") or (
            (not self.is_decoder) and self.config.encoder_adapter == "task"
        ):
            self.param_gen = ParameterGenerator(
                config, config.hidden_size, is_encoder=not self.is_decoder
            )
            self.adapter_task_embedding = nn.Embedding(
                len(self.config.tasks), self.config.d_model
            )

    def forward(
        self,
        input_ids=None,   # decoder 的标签
        encoder_hidden_states=None,
        tasks=None,
        encoder_attention_mask=None,
        context_for_hyper_embedding=None,  # 关于 question 在  context上查询的编码结果，即交叉注意力
        context_for_hyper_mask=None,
        **kwargs,
    ):
        # using input ids to determine whats going
        self.clear_adapters()
        if self.is_decoder and self.config.decoder_adapter == "generated":
            # todo: 专家模块层，这里可以存储知识，可以根据当前context的编码k-v 查询出 lora 相关的知识
            # other_knowledge_for_decoder_adapter = self.experts_for_context(org_context_states, residual=org_context_states)
            mean_hidden_states = mean_pooling(context_for_hyper_embedding, context_for_hyper_mask)  # todo: 编码器的结果 取平均得到只有一个token长度
            # mean_other_knowledge_hidden_states = mean_pooling(other_knowledge_for_decoder_adapter, org_context_states_mask)
            out_res = self.param_gen(
                self.mlp(mean_hidden_states),
                # other_context_knowledge=self.mlp_experts(mean_other_knowledge_hidden_states)
            )
            self.apply_params_to_adapters(       # todo: 直接将 ”encoder经过超网络的输出“ ——> 赋值为 adapter中矩阵的权重
                context_for_hyper_embedding.size(0),
                out_res,
                # out_res_other
            )
        elif (not self.is_decoder) and self.config.encoder_adapter == "generated":
            # for encoder generation, we first pass through the encoder, then set encoder adapters based on this.
            # currently using learnt adapters in the first pass, but potentially we could turn those off too?
            res = super().forward(
                input_ids=input_ids,
                encoder_hidden_states=encoder_hidden_states,
                **kwargs,
            )
            self.apply_params_to_adapters(
                input_ids.size(0),
                self.param_gen(
                    self.mlp(
                        mean_pooling(res.last_hidden_state, kwargs["attention_mask"])
                    )
                ),
            )
        elif (self.is_decoder and self.config.decoder_adapter == "task") or (
            not self.is_decoder and self.config.encoder_adapter == "task"
        ):
            # at test time, we only test one task at a time.
            if not self.training:
                # simple sanity check
                if len(tasks) > 0:
                    assert(tasks[0] == tasks[1] and tasks[1] == tasks[-1])
                tasks = [tasks[0] for _ in range(input_ids.size(0))]
            indices = torch.tensor(
                [self.config.tasks.index(task) for task in tasks],
                device=input_ids.device,
                dtype=torch.long,
            )
            task_embed = self.adapter_task_embedding(indices)
            self.apply_params_to_adapters(input_ids.size(0), self.param_gen(task_embed))
        elif (self.is_decoder and self.config.decoder_adapter == "manual_specific") or (
            not self.is_decoder and self.config.encoder_adapter == "manual_specific"
        ):
            indices = torch.tensor(
                [self.config.tasks.index(task) for task in tasks],
                device=input_ids.device,
                dtype=torch.long,
            )
            self.apply_indices_to_adapters(indices)
        return super().forward(   # todo: encoder 和 decoder 都会执行这里
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask, **kwargs
        )

    def clear_adapters(self):
        for block in self.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerFFWithAdapter):
                    layer.adapter_layer.clear_adapter()

    def apply_params_to_adapters(self, batch_size, generated_params):
        for param, block in zip(generated_params, self.block):
            block.layer[-1].adapter_layer.apply_adapter_params(batch_size, *param)

    def apply_indices_to_adapters(self, indices):
        for block in self.block:
            block.layer[-1].adapter_layer.set_indices(indices)

import pytorch_lightning as pl
class T5ForConditionalGenerationWithAdapterWithFusion_WithVip(T5ForConditionalGeneration, pl.LightningModule):
    def __init__(self, config):
        super().__init__(config)
        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5StackWithAdapter(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5StackWithAdapter(decoder_config, self.shared)

        Gated_Cross_Attention_config = {}
        for key2, value2 in zip(["num_xattn_layers", "cross_attn_every", "dim", "dim_head", "heads", "xattn_ff_mult", "only_attend_immediate_media"],
                              [config.num_xattn_layers, config.cross_attn_every, config.dim, config.dim_head,
                               config.heads, config.xattn_ff_mult, config.only_attend_immediate_media]):
            Gated_Cross_Attention_config.update({key2: value2})
        num_xattn_layers = Gated_Cross_Attention_config["num_xattn_layers"]
        cross_attn_every = Gated_Cross_Attention_config["cross_attn_every"]
        # xattn blocks (only use one block for now)
        if num_xattn_layers == 1:
            ### one xattn one self-atnn
            self.xattn_stack = nn.ModuleList()
            self.xattn_stack.append(
                GatedCrossAttentionBlock(    # todo:
                    dim=Gated_Cross_Attention_config["dim"],
                    dim_head=Gated_Cross_Attention_config["dim_head"],
                    heads=Gated_Cross_Attention_config["heads"],
                    ff_mult=Gated_Cross_Attention_config["xattn_ff_mult"],
                    only_attend_immediate_media=Gated_Cross_Attention_config["only_attend_immediate_media"]
                )
        )
        elif num_xattn_layers > 1:
            inserting_xattn_positions = list(range(0, num_xattn_layers, cross_attn_every))
            print("inserting_xattn_positions:", inserting_xattn_positions)
            layers = [T5Block(encoder_config, has_relative_attention_bias=False) for i in range(num_xattn_layers)]

            for i, ins_i in enumerate(inserting_xattn_positions):
                layers.insert(ins_i + i, GatedCrossAttentionBlock(
                    dim=Gated_Cross_Attention_config["dim"],
                    dim_head=Gated_Cross_Attention_config["dim_head"],
                    heads=Gated_Cross_Attention_config["heads"],
                    ff_mult=Gated_Cross_Attention_config["xattn_ff_mult"],
                    only_attend_immediate_media=Gated_Cross_Attention_config["only_attend_immediate_media"]
                ))
            print("xattn layers:", layers)
            self.xattn_stack = nn.ModuleList(layers)

        # 3. soft prompt 实现 vip
        self.prompt_config = copy.deepcopy(config)
        self.prompt_embedding = nn.Parameter(torch.empty((self.prompt_config.num_cq_tokens, self.prompt_config.d_model), dtype=torch.float))
        self.vip_fc_in = nn.Linear(self.prompt_config.d_model, self.prompt_config.project_dim, bias=True)
        self.vip_fc_out = nn.Linear(self.prompt_config.project_dim, self.prompt_config.d_model, bias=True)
        self.vip_encoder_layer = TransformerEncoderLayer(d_model=self.prompt_config.project_dim, nhead=self.prompt_config.trans_attention_head, dim_feedforward=config.trans_linear_dim)  # 2048
        self.sentence_encoder = TransformerEncoder(self.vip_encoder_layer, num_layers=self.prompt_config.trans_num_layers)
        self.codebook = nn.Embedding.from_pretrained(
            self.shared.weight[self.prompt_config.num_cq_tokens:self.prompt_config.num_cq_tokens + self.prompt_config.codebook_size].clone().detach(),
            freeze=False)
        self.register_buffer('_ema_cluster_size', torch.ones(self.prompt_config.codebook_size) / self.prompt_config.num_codes)
        # init prompt_embedding + codebook
        self.codebook.weight.data = 100 * torch.nn.functional.normalize(self.codebook.weight.data, dim=-1)
        nn.init.normal_(self.prompt_embedding.data, mean=0.0, std=config.initializer_factor * 1.0)

        self._initialize_linear_weights()  # init newly added parameters
        # self.init_weights()
        self.post_init()

    def _initialize_linear_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
        self.apply(init_weights)
        print('initialize nn.Linear with xavier_normal_')
    # todo: 运行 generate 时，会首先获取编码器的输出
    def get_encoder(self):
        return self.encode_context_and_question

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        tasks=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        aug_input_ids=None,  # 这个参数是传入 context的编码
        aug_attention_mask=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encode_context_and_question(   # 自己定义的 encode, 包括两部分，一部分是 编码 context, 另一部分是编码question
                input_ids=input_ids,
                attention_mask=attention_mask,
                tasks=tasks,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                aug_input_ids=aug_input_ids,
                aug_attention_mask=aug_attention_mask
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # hidden_states = encoder_outputs.last_hidden_state  # 对于 question的编码  (8,56,1024)
        context_for_hyper_input = encoder_outputs.question_query_to_context_hidden_state  # 对于 question和 context互注意力的编码，用于输入超网络产生adapter参数 (8,56,1024)
        # org_context_states = encoder_outputs.org_context_hidden_state  # 对于 context的编码 (8,56*2,1024)
        with_prompt_hidden_state = encoder_outputs.with_prompt_hidden_state  # 对于context + prompt + question的编码(8, seq_len, 1024)
        with_prompt_attention_mask = encoder_outputs.with_prompt_attention_mask
        with_prompt_aux_loss = encoder_outputs.aux_loss  # prompt时需要的损失

        if aug_attention_mask is None:  # 1. aug_attention_mask训练时来源自 模型输入
            aug_attention_mask = encoder_outputs.org_context_attention_mask
            aug_attention_mask = rearrange(aug_attention_mask, '(b m) n -> b m n', b=attention_mask.size(0))
            aug_attention_mask = rearrange(aug_attention_mask, 'b m n -> b (m n)')
        else:  # 2. 测试时 来源自 encode_context_and_question的返回值
            aug_attention_mask = rearrange(aug_attention_mask, 'b m n -> b (m n)')

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            tasks=tasks,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=with_prompt_hidden_state,  # decoder的输入由原来只是 question的编码--> context +prompt + question的编码
            encoder_attention_mask=with_prompt_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context_for_hyper_embedding=context_for_hyper_input,
            context_for_hyper_mask=attention_mask  # 超网络的输入，用来构造decoder的 adapter，
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
            # todo: 交叉熵loss + aux_loss
            loss += with_prompt_aux_loss  # 只有在 训练时计算loss

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output


        return Seq2SeqLMOutput_with_two_loss(
            aux_loss=with_prompt_aux_loss,
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


    def _prepare_encoder_decoder_kwargs_for_generation(
        self, inputs_tensor: torch.Tensor, model_kwargs, model_input_name: Optional[str] = None
    ) -> Dict[str, Any]:
        # 1. get encoder
        encoder = self.get_encoder()
        # Compatibility with Accelerate big model inference: we need the encoder to outputs stuff on the same device
        # as the inputs.
        if hasattr(encoder, "_hf_hook"):
            encoder._hf_hook.io_same_device = True

        # 2. Prepare encoder args and encoder kwargs from model kwargs.
        irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
        encoder_kwargs = {
            argument: value
            for argument, value in model_kwargs.items()
            if not any(argument.startswith(p) for p in irrelevant_prefix)
        }
        # todo: 我修改的这里
        encoder_kwargs = {
            argument: value for argument, value in encoder_kwargs.items()
        }

        # 3. make sure that encoder returns `ModelOutput`
        model_input_name = model_input_name if model_input_name is not None else self.main_input_name
        encoder_kwargs["return_dict"] = True
        encoder_kwargs[model_input_name] = inputs_tensor
        model_kwargs["encoder_outputs"]: ModelOutput = encoder(**encoder_kwargs)

        return model_kwargs

    # required to pass tasks through
    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            decoder_attention_mask=None,
            cross_attn_head_mask=None,
            use_cache=None,
            encoder_outputs=None,   # 这里包含了关于输入 编码的一些， 包含关于问题和 context的编码，
            **kwargs,
    ):
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache
        }

    def encode_context_and_question(self,
                                    input_ids=None,
                                    attention_mask=None,
                                    tasks=None,
                                    inputs_embeds=None,
                                    head_mask=None,
                                    output_attentions=None,
                                    output_hidden_states=None,
                                    return_dict=None,
                                    aug_input_ids=None,
                                    aug_attention_mask=None,
                                    aug_encoder_outputs=None,
                                    **kwargs,
                                    ):

        # todo: 1-1. encode input---------针对question的编码结果
        input_encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            tasks=tasks,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # todo: 1-2. encode augmentations individually-------针对 context证据的编码
        if aug_encoder_outputs is None and aug_input_ids is not None:
            # get hidden states
            if aug_input_ids.ndim == 2:
                assert aug_attention_mask.ndim == 2
                aug_input_ids = rearrange(aug_input_ids, 'b n -> b 1 n')
                aug_attention_mask = rearrange(aug_attention_mask, 'b n -> b 1 n')

            b, m, n = aug_input_ids.shape  # 其中m代表一个样本拥有的context 数量
            ## may cause CUDA OOM?:
            aug_input_ids = rearrange(aug_input_ids, 'b m n -> (b m) n')  # group into a larger batch
            aug_attention_mask = rearrange(aug_attention_mask, 'b m n -> (b m) n')

            aug_encoder_outputs = self.encoder(
                input_ids=aug_input_ids, attention_mask=aug_attention_mask, return_dict=True)
            aug_encoder_outputs = aug_encoder_outputs.last_hidden_state  # (b m) n d
            aug_encoder_outputs = rearrange(aug_encoder_outputs, '(b m) n d -> b (m n) d', b=b)   # m 代表每个问题有几个context, n 是每个context的长度
            # 针对问题的编码
            xattn_hidden_states = input_encoder_outputs[0]  # encoded input, question 编码

            input_shape = input_ids.size()
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
            # todo:  1.3 question|context --------执行交叉注意力
            for i, layer_module in enumerate(self.xattn_stack):  # todo: 执行 "Grated Cross Attention" 模块
                if isinstance(layer_module, GatedCrossAttentionBlock):
                    xattn_hidden_states = layer_module(
                        xattn_hidden_states,
                        aug_encoder_outputs,
                        aug_exist_idx=None
                    )
                elif isinstance(layer_module, T5Block):
                    xattn_hidden_states = layer_module(
                        xattn_hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=None
                    )[0]
                else:
                    raise NotImplementedError

            # todo: 1.4 context +prompt + question -------联合输入，构造样本级 prompt，
            with_prompt_inputs_embeds, with_prompt_attention_mask, aux_loss = self.prompt_forward(input_ids, attention_mask, aug_input_ids, aug_attention_mask)
            # 将布尔值转换为 int64 类型
            with_prompt_attention_mask = torch.tensor(with_prompt_attention_mask, dtype=torch.int64)
            with_prompt_input_encoder_output = self.encoder(
                attention_mask=with_prompt_attention_mask,
                inputs_embeds=with_prompt_inputs_embeds,
                return_dict=True,
            )
            with_prompt_hidden_state = with_prompt_input_encoder_output.last_hidden_state

        encoder_outputs = BaseModelOutput_T5(
            # last_hidden_state=new_tmp if len(need_replaced_position) !=0 else xattn_hidden_states,
            question_query_to_context_hidden_state=xattn_hidden_states,      # 问题和context交叉注意力后的输出，
            last_hidden_state=input_encoder_outputs[0],  # 针对问题的编码 输出，
            org_context_hidden_state=aug_encoder_outputs,  # context 只经过 encoder的输出
            org_context_attention_mask=aug_attention_mask,
            with_prompt_hidden_state=with_prompt_hidden_state,
            with_prompt_attention_mask=with_prompt_attention_mask,
            hidden_states=None,
            attentions=None,
            aux_loss=aux_loss
        )
        return encoder_outputs

    def prompt_forward(self, input_ids, attention_mask, aug_input_ids, aug_attention_mask):
        # Concatenate the prefix embeddings and extend the attention masks
        batch_size = input_ids.size(0)
        device = input_ids.device
        aug_input_ids = rearrange(aug_input_ids, '(b m) n -> b m n', b=batch_size)
        aug_input_ids = rearrange(aug_input_ids, 'b m n -> b (m n)')
        aug_attention_mask = rearrange(aug_attention_mask, '(b m) n -> b m n', b=batch_size)
        aug_attention_mask = rearrange(aug_attention_mask, 'b m n -> b (m n)')
        inputs_embeds = self.shared(input_ids)  #
        aug_input_embeds = self.shared(aug_input_ids)
        prompt_embeds = self.prompt_embedding[None, :, :].expand(batch_size, -1, -1).to(device)  # (100, 1024) -> (8, 100, 1024)
        # 构造输入的形式如下：context prompt question, prompt在中间，沟通证据和问题
        inputs_embeds = torch.cat([aug_input_embeds, prompt_embeds, inputs_embeds], dim=1)
        prompt_attention_mask = torch.ones((batch_size, self.prompt_config.num_cq_tokens), dtype=attention_mask.dtype).to(device)
        attention_mask = torch.cat([aug_attention_mask, prompt_attention_mask, attention_mask], dim=1).bool()

        xc_down = self.vip_fc_in(inputs_embeds)
        xc_down = xc_down.transpose(0, 1)  # batch * seq_len * 64-> (aug_len + prompt_len + input_len) *  batch * 64
        xc_up = self.sentence_encoder(src=xc_down, src_key_padding_mask=~attention_mask).transpose(0, 1)
        start = aug_attention_mask.size(1)
        end = start + self.prompt_config.num_cq_tokens

        xc_out = self.vip_fc_out(xc_up[:, start:end, :])
        xc_out_shape = xc_out.size()  # batch * prompt_num * dim
        # Flatten input
        flat_xc_out = xc_out.reshape(-1, self.prompt_config.d_model)  # (batch* prompt_len) * dim
        # calculate distances
        distances = (torch.sum(flat_xc_out ** 2, dim=1, keepdim=True)
                     + torch.sum(self.codebook.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_xc_out, self.codebook.weight.t()))  # 返回尺寸是 (batch*prompt_len) * code_size

        # Define multinomial distribution and sample from it,
        temperature = self.prompt_config.temperature
        multi = Multinomial(total_count=self.prompt_config.num_samples, logits=(-distances - 1e-5) / temperature)
        samples = multi.sample().to(device)    # (batch*prompt_len) * code_size, 每行只有10个位置是，其他位置是0
        nu_samples = samples.cpu().numpy()
        # Soft-quantize and unflatten，----> (batch * prompt_len) * dim --> batch * prompt_len * dim
        xc_quantized = torch.matmul(samples, self.codebook.weight).view(xc_out_shape) / self.prompt_config.num_samples

        # Loss
        e_latent_loss = torch.mean((xc_quantized.detach() - xc_out) ** 2)
        loss = self.prompt_config.commitment_cost * e_latent_loss

        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self.prompt_config._decay + \
                                     (1 - self.prompt_config._decay) * \
                                     (torch.sum(samples, 0) / self.prompt_config.num_samples)  # codebook_size

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self.prompt_config._epsilon)
                    / (n + self.prompt_config.codebook_size * self.prompt_config._epsilon) * n)

            dw = torch.matmul(samples.t(), flat_xc_out) / self.prompt_config.num_samples  # codebook_size * dim
            normalized_ema_w = self.codebook.weight * self.prompt_config._decay + (1 - self.prompt_config._decay) * (
                        dw / self._ema_cluster_size.unsqueeze(1))  # option-1
            self.codebook.weight = nn.Parameter(normalized_ema_w)

        xc_quantized = xc_out + (xc_quantized - xc_out).detach()  # todo: 原始的静态prmopt输出 + 量化的prompt
        outputs_embeds = inputs_embeds.clone()
        outputs_embeds[:, start:end, :] = xc_quantized
        # loss[loss <= 0.0009] = 0.0  # 如果减少 这个对于总loss的影响呢
        # loss.fill_(0.0)
        return outputs_embeds, attention_mask, loss

from dataclasses import dataclass
@dataclass
class BaseModelOutput_T5(BaseModelOutput):
    """
    Base class for model's outputs, with potential hidden states and attentions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    question_query_to_context_hidden_state: torch.FloatTensor = None
    org_context_hidden_state: torch.FloatTensor = None    # context 只经过 encoder的输出
    org_context_attention_mask: torch.FloatTensor = None    # context 的掩码

    with_prompt_hidden_state: torch.FloatTensor = None
    with_prompt_attention_mask: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None

from dataclasses import dataclass
@dataclass
class Seq2SeqLMOutput_with_two_loss(Seq2SeqLMOutput):
   aux_loss: Optional[torch.FloatTensor] = None
