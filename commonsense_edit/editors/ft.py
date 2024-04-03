import torch
import pytorch_lightning as pl
import transformers
from transformers import get_linear_schedule_with_warmup
import sys
sys.path.append("/media/data/1/yx/code/edit_knowledge_patch")
import commonsense_edit.modeling.modeling_t5xl as trans
from commonsense_edit.utils import freeze_most_layer

class Finetune(pl.LightningModule):
    def __init__(self, config=None, step=None):
        super(Finetune, self).__init__()
        if 'xl' in config.model.model_cache:
            ModelClass = getattr(trans, 'T5ForConditionalGeneration')
        else:
            try:
                ModelClass = getattr(transformers, config.model.model_class)
            except:
                ModelClass = getattr(trans, 'T5ForConditionalGeneration')
        print(f"Loading model class {ModelClass} from cache dir {config.model.model_cache}")
        self.model = ModelClass.from_pretrained(config.model.model_cache)
        if 'xl' in config.model.model_cache:
            freeze_most_layer(self.model, config)
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print("Parameter name %s", name)

        Tokenzir_Class = getattr(transformers, config.model.tokenizer_class)
        self.tokenizer = Tokenzir_Class.from_pretrained(config.model.model_cache)

        self.config = config
        self.learning_rate = config.model.fine_tune.learning_rate
        self.step = step
        self.data_size = config.data_size if 'data_size' in config else 0
        self.epoch = config.model.fine_tune.n_epochs if 'n_epochs' in config.model.fine_tune else 0
        self.warmup_step = config.warmup_step if 'warmup_step' in config else 0

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