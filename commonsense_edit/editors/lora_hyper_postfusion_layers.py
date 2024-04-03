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
                                   unfreeze_perceiver_resampler_layers_
from commonsense_edit.modeling.adapter_t5 import T5WithAdapterConfig, T5ForConditionalGenerationWithAdapter
from commonsense_edit.modeling.adapter_fusion_layers_t5 import T5ForConditionalGenerationWithAdapterWithFusionOneLayer


class Lora_Postfusion_Layers(pl.LightningModule):
    def __init__(self, config=None, step=None):
        super(Lora_Postfusion_Layers, self).__init__()
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
            unfreeze_adapter_params_decoder(self.model, config)

        if model_config.unfreeze_gated_and_perceiver:
            # unfreeze_gated_cross_attention_layers_(self.model)
            unfreeze_perceiver_resampler_layers_(self.model)

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Parameter name %s", name)

        # 计算需要训练的参数数量
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        Tokenzir_Class = getattr(transformers, config.model.tokenizer_class)
        self.tokenizer = Tokenzir_Class.from_pretrained(config.model.model_cache)

        self.config = config
        self.learning_rate = config.model.fine_tune.learning_rate
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

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        context = batch['context_input_ids']
        context_attention_mask = batch['context_attention_mask']

        loss, output = self(input_ids, attention_mask, labels, context=context, context_attention_mask=context_attention_mask)
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
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_step,
            num_training_steps=self.epoch*self.data_size,
        )
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]