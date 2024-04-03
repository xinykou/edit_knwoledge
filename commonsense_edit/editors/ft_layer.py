import torch
import pytorch_lightning as pl
import transformers
from transformers import get_linear_schedule_with_warmup
from commonsense_edit.utils import brackets_to_periods

class Finetune_Layer(pl.LightningModule):
    def __init__(self, config=None, step=None):
        super(Finetune_Layer, self).__init__()

        ModelClass = getattr(transformers, config.model.model_class)
        print(f"Loading model class {ModelClass} from cache dir {config.model.model_cache}")
        self.model = ModelClass.from_pretrained(config.model.model_cache)
        Tokenzir_Class = getattr(transformers, config.model.tokenizer_class)
        self.tokenizer = Tokenzir_Class.from_pretrained(config.model.model_cache)

        self.config = config
        self.learning_rate = config.model.fine_tune.learning_rate
        self.step = step
        self.data_size = config.data_size if 'data_size' in config else 0
        self.epoch = config.model.fine_tune.n_epochs if 'n_epochs' in config.model.fine_tune else 0
        self.warmup_step = config.warmup_step if 'warmup_step' in config else 0

        self.pnames = brackets_to_periods(config.editor.inner_params[0])
        for n, p in self.model.named_parameters():
            if n != self.pnames:
                p.requires_grad = False
            else:
                p.requires_grad = True

        # 计算需要训练的参数数量
        total_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return output.loss, output.logits

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, output = self.forward(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss

    # def validation_end(self, outputs):
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     if avg_loss < self.config.stop_loss_value:
    #         self.trainer.should_stop = True


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