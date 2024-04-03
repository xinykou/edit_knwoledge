import torch
import pytorch_lightning as pl
import transformers
from transformers import get_linear_schedule_with_warmup
import sys
sys.path.append('/media/data/1/yx/code/edit_knowledge_patch')

from commonsense_edit.utils import freeze_params, \
                                   unfreeze_adapter_params_encoder, \
                                   unfreeze_adapter_params_decoder, \
                                   unfreeze_gated_cross_attention_layers_, \
                                   unfreeze_perceiver_resampler_layers_ , \
                                    unfreeze_experts_layers_

from commonsense_edit.modeling.adapter_t5 import T5WithAdapterConfig
from commonsense_edit.modeling.adapter_fusion_experts_t5 import T5ForConditionalGenerationWithAdapterWithFusion_Experts
import torch.nn.functional as F

class Lora_Postfusion_Expert(pl.LightningModule):
    def __init__(self, config=None, step=None):
        super(Lora_Postfusion_Expert, self).__init__()
        ConfigClass = globals()[config.model.config_class]
        model_config = ConfigClass.from_pretrained(config.model.model_cache)  # 跟模型相关的所有参数
        model_config.update(config.editor)  # 新添加的参数

        ModelClass = globals()[config.model.model_class]
        print(f"Loading model class {ModelClass} from cache dir {config.model.model_cache}")
        self.model = ModelClass.from_pretrained(config.model.model_cache, config=model_config)
        # todo: 模型冻结
        if model_config.freeze_model:
            freeze_params(self.model)
        if model_config.unfreeze_encoder_adapters:
            unfreeze_adapter_params_encoder(self.model)
        if model_config.unfreeze_decoder_adapters:
            unfreeze_adapter_params_decoder(self.model)

        if model_config.unfreeze_gated_and_perceiver:
            unfreeze_gated_cross_attention_layers_(self.model)
            unfreeze_perceiver_resampler_layers_(self.model)

        if model_config.unfreeze_experts:
            unfreeze_experts_layers_(self.model, config.editor)  # 包含是否使用 random expert的策略


        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Parameter name %s", name)

        Tokenzir_Class = getattr(transformers, config.model.tokenizer_class)
        self.tokenizer = Tokenzir_Class.from_pretrained(config.model.model_cache)

        self.config = config
        self.learning_rate_main = config.model.fine_tune.learning_rate_main
        self.learning_rate_other = config.model.fine_tune.learning_rate_other
        self.step = step
        self.data_size = config.data_size if 'data_size' in config else 0
        self.epoch = config.model.fine_tune.n_epochs if 'n_epochs' in config.model.fine_tune else 0
        self.warmup_step = config.warmup_step if 'warmup_step' in config else 0

    def forward(self, input_ids, attention_mask, labels=None, context=None, context_attention_mask=None):
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                            aug_input_ids=context,
                            aug_attention_mask=context_attention_mask
                            )
        return output.loss, output.logits


    # 当启动随机选择一个 expert时
    def symmetric_KL_loss(self, p, q, attention_mask, reduction='batchmean'):
        """ symmetric KL-divergence 1/2*(KL(p||q)+KL(q||p)) """
        p, q, attention_mask = p.float(), q.float(), attention_mask.view(-1)
        dict_size = q.size(-1)
        attention_mask = attention_mask.eq(1.0)
        p = p.view(-1, dict_size)[attention_mask]
        q = q.view(-1, dict_size)[attention_mask]

        loss = F.kl_div(F.log_softmax(p, dim=-1, dtype=torch.float32),
                        F.softmax(q.detach(), dim=-1, dtype=torch.float32), reduction=reduction) + \
               F.kl_div(F.log_softmax(q, dim=-1, dtype=torch.float32),
                        F.softmax(p.detach(), dim=-1, dtype=torch.float32), reduction=reduction)
        return 0.5 * loss.mean()

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        context = batch['context_input_ids']
        context_attention_mask = batch['context_attention_mask']

        loss, logits = self(input_ids, attention_mask, labels, context=context, context_attention_mask=context_attention_mask)

        if self.config.editor.using_random_experts:  # 每次是否只随机 选择一个 experts
            lm_logits = logits
            labels_mask = batch["labels_mask"]
            _, logits_second = self(input_ids, attention_mask, labels, context=context, context_attention_mask=context_attention_mask)
            lm_logits_additional = logits_second
            consistency_loss = self.symmetric_KL_loss(lm_logits, lm_logits_additional, labels_mask) * self.config.editor.kl_weight
            loss += consistency_loss

        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        context = batch['context_input_ids']
        context_attention_mask = batch['context_attention_mask']

        loss, output = self(input_ids, attention_mask, labels, context=context, context_attention_mask=context_attention_mask)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        context = batch['context_input_ids']
        context_attention_mask = batch['context_attention_mask']

        loss, output = self.forward(input_ids, attention_mask, labels, context=context, context_attention_mask=context_attention_mask)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        # print(self.parameters())
        all_params = self.named_parameters()   # 返回名字+参数
        params_main = [p for name, p in all_params if not 'experts_for_context' in name]
        params_other = [p for name, p in all_params if 'experts_for_context' in name]
        optimizer = torch.optim.Adam([
            {'params': params_main, 'lr': self.learning_rate_main},
            {'params': params_other, 'lr': self.learning_rate_other}
        ])

        # optimizer = torch.optim.AdamW(
        #     self.parameters(),
        #     lr=self.learning_rate,
        # )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_step,
            num_training_steps=self.epoch*self.data_size,
        )
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]