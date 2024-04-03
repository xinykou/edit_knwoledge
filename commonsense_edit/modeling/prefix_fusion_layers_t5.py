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
from commonsense_edit.modeling.modeling_t5 import (
    T5Block,
    T5LayerFF,
    T5Stack,
    T5ForConditionalGeneration,
    __HEAD_MASK_WARNING_MSG,
)

from .adapter_generators import ParameterGenerator
from .adapter_layer import AdapterLayer, TaskSpecificAdapterLayer, PrefixLayer
from .xattn_layer import GatedCrossAttentionBlock, PerceiverResampler, CrossAttentionBlock
from .xattn_layer import freeze_all_layers_, unfreeze_all_layers_
from .xattn_layer import Classify_for_adapter_Weight





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

class T5LayerFFWithPrefix(T5LayerFF):
    def __init__(self, config, is_encoder=False):
        super().__init__(config)
        self.config = config
        self.prefix_layer = PrefixLayer(config)
        self.num_prompt_tokens = config.num_prompt_tokens

    def forward(self, hidden_states):
        normed_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(normed_states)
        adapter_input = normed_states + forwarded_states
        prompt_embedding = self.prefix_layer()
        hidden_states = torch.cat([prompt_embedding, adapter_input], dim=1)
        return hidden_states

class T5BlockWithPrefix(T5Block):
    def __init__(self, config, pos=None, decoder=False, has_relative_attention_bias=False, is_encoder=False):
        super().__init__(
            config, has_relative_attention_bias=has_relative_attention_bias
        )
        if getattr(config, 'only_using_last_layers', None) ==True and decoder:
            if pos:
                self.layer[-1] = T5LayerFFWithPrefix(config, is_encoder=is_encoder)
            else:
                pass
        # 如果不关心最后一层
        else:
            self.layer[-1] = T5LayerFFWithAdapter(config, is_encoder=is_encoder)


def mean_pooling(hidden_state, attention_mask=None):
    input_masked = hidden_state * attention_mask.unsqueeze(-1)
    return input_masked.sum(1) / attention_mask.sum(1).unsqueeze(-1)

# encoder 或者 decoder
class T5StackWithAdapter(T5Stack):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config, embed_tokens=embed_tokens)
        blockClass = T5Block
        self.config = config
        if (self.is_decoder and self.config.decoder_adapter != "none") or (
            (not self.is_decoder) and self.config.encoder_adapter != "none"
        ):
            blockClass = T5BlockWithPrefix   # todo: 执行这里，相当于在ffn层 部分加了 adapter
            kwargs = {"is_encoder": not self.is_decoder}
        else:
            kwargs = {}
        self.block = torch.nn.ModuleList(   # 每一层 拼接在一起，形成 编码器 或者解码器
            [
                blockClass(config, pos=True if i-config.num_layers == config.inject_layers_pos else False, decoder=self.is_decoder, has_relative_attention_bias=bool(i == 0), **kwargs)
                for i in range(config.num_layers)

            ]
        )
        if (self.is_decoder and self.config.decoder_adapter == "generated") or (
            (not self.is_decoder) and self.config.encoder_adapter == "generated"
        ):
            # self.classifier = Classify_for_adapter_Weight()
            if getattr(config, 'only_using_last_layers', None) == True:
                only_using_last_layer = True
            else:
                only_using_last_layer = False

            self.param_gen = ParameterGenerator(        # todo: 这里是用来 使用超网络生成 解码器需要的参数
                config, config.hidden_size, is_encoder=not self.is_decoder, only_using_last_layer=only_using_last_layer
            )
            if self.config.process_encoder_output:
                self.mlp = nn.Sequential(
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
        input_ids=None,
        encoder_hidden_states=None,
        tasks=None,
        context_for_hyper_input=None,
        classify_hidden_states=None,
        **kwargs,
    ):
        # using input ids to determine whats going
        self.clear_prefix()
        if self.is_decoder and self.config.decoder_adapter == "generated":

            # if len(classify_hidden_states.shape) == 4:  # media 是 preceiver resampler输出的维度，多了一个time，是 因为为各个证据建立了不同的标志位
            #     media = rearrange(classify_hidden_states, 'b t m d -> b (t m) d')  # 将新添加的维度 合并回去

            self.apply_params_to_adapters(       # todo: 直接将 encoder经过超网络的输出 生成 adapter中矩阵的权重
                context_for_hyper_input.size(0),
                self.mlp(context_for_hyper_input),
                inject_layers=self.config.inject_layers_pos  # 超网络注入到decoder哪一层
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
        return super().forward(
            input_ids=input_ids, encoder_hidden_states=encoder_hidden_states, **kwargs
        )

    def clear_prefix(self):
        for block in self.block:
            for layer in block.layer:
                if isinstance(layer, T5LayerFFWithPrefix):
                    layer.prefix_layer.clear_prefix()

    def apply_params_to_adapters(self, batch_size, generated_params, inject_layers=None):
        if inject_layers != None:  # todo: 说明只生成了最后一层的参数
            self.block[inject_layers].layer[-1].prefix_layer.apply_prefix_params(feature=generated_params)

        else:
            for param, block in zip(generated_params, self.block):
                block.layer[-1].adapter_layer.apply_adapter_params(batch_size, *param)

    def apply_indices_to_adapters(self, indices):
        for block in self.block:
            block.layer[-1].adapter_layer.set_indices(indices)

import pytorch_lightning as pl
class T5ForConditionalGenerationWithPrefixWithFusionOneLayer(T5ForConditionalGeneration, pl.LightningModule):
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

        perceiver_config = {}
        for key1, value1 in zip(["dim", "depth", "dim_head", "heads", "num_latents", "num_aug_sources", "ff_mult"],
                              [config.dim, config.depth, config.dim_head, config.heads, config.num_latents, config.num_aug_sources, config.ff_mult]):
            perceiver_config.update({key1: value1})
        self.perceiver_resampler = PerceiverResampler(**perceiver_config)  # todo:

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

        self._initialize_linear_weights()  # init newly added parameters
        # self.init_weights()
        self.post_init()

    def _initialize_linear_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
        self.apply(init_weights)
        print('initialize nn.Linear with xavier_normal_')

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

        hidden_states = encoder_outputs.last_hidden_state  # 对于 question的编码
        context_for_hyper_input = encoder_outputs.question_query_to_context_hidden_state  # 对于 question和 context互注意力的编码，用于输入超网络产生adapter参数
        classify_hidden_states = encoder_outputs.org_context_hidden_state
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
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            context_for_hyper_input=context_for_hyper_input,  # 超网络的输入，用来构造decoder的 adapter，
            classify_hidden_states=classify_hidden_states
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

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
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
            encoder_outputs=None,
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

        # todo: 1-1. encode input
        input_encoder_outputs = self.encoder(        # 针对question的编码结果
            input_ids=input_ids,
            attention_mask=attention_mask,
            tasks=tasks,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # todo：1-2. encode augmentations individually      # 针对 context证据的编码
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
            aug_encoder_outputs = rearrange(aug_encoder_outputs, '(b m) n d -> b m n d', b=b)   # m 代表每个问题有几个context, n 是每个context的长度

            xattn_hidden_states = input_encoder_outputs[0]  # encoded input, question 编码
            # need_replaced_position = []
            # new_tmp = xattn_hidden_states.clone()
            # aug_input_ids_four = rearrange(aug_input_ids, '(b m) n -> b m n', b=b)
            # for index in range(input_ids.size(0)):
            #     if not torch.equal(input_ids[index], aug_input_ids_four[index][0]):
            #         need_replaced_position.append(index)

            # feed into resampler to get resampled concatenated augmentation vector
            # classify_hidden_states = self.perceiver_resampler(aug_encoder_outputs)  # todo: 执行 ”Perceiver Resampleer“ 模块: batch, m, n, d,
            # aug_hidden_states = aug_encoder_outputs

            ### interleaved xattn and self attn
            input_shape = input_ids.size()
            extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)
            # todo: 1.3 question|context 执行交叉注意力
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

        encoder_outputs = BaseModelOutput_T5(
            # last_hidden_state=new_tmp if len(need_replaced_position) !=0 else xattn_hidden_states,
            question_query_to_context_hidden_state=xattn_hidden_states,  # 问题和context交叉注意力后的输出，
            last_hidden_state=input_encoder_outputs[0],
            org_context_hidden_state=aug_encoder_outputs,  # context 只经过 encoder的输出
            org_context_attention_mask=aug_attention_mask,
            hidden_states=None,
            attentions=None
        )
        return encoder_outputs


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
    org_context_hidden_state: torch.FloatTensor = None  # context 只经过 encoder的输出
    org_context_attention_mask: torch.FloatTensor = None  # context 的掩码