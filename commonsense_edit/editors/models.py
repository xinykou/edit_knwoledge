import torch
import logging
import transformers
from torch import nn
import os
from .ft import Finetune
from .grace import GRACE

LOG = logging.getLogger(__name__)

def pretrain(model, loader, tokenize, n_epochs, device):
    optimizer = torch.optim.Adam(model.model.parameters(), lr=1e-4)
    for _ in range(n_epochs):
        losses = []
        for batch in loader:
            batch = tokenize(batch, model.tokenizer, device)
            loss = model.model(**batch).loss
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            model.model.zero_grad()
    return model

def get_tokenizer(config):
    return getattr(transformers, config.model.tokenizer_class).from_pretrained(config.model.model_cache)

def get_hf_model(config):

    ModelClass = globals()["Finetune"]

    if config.train_mode == "train":
        checkpoint_path = os.path.join(config.model.fine_tune.checkpoints_dirpath, config.checkpoint_filename)
        LOG.info(f"Loading model class {ModelClass} from cache dir {checkpoint_path}")
        model = ModelClass.load_from_checkpoint(checkpoint_path, strict=False, config=config)  # pytorch lightning 的加载方式
        return model.model

    elif config.train_mode == "test":
        checkpoint = torch.load(os.path.join(config.model.fine_tune.checkpoints_dirpath, config.checkpoint_filename))
        model = ModelClass(config=config).model
        model.load_state_dict(checkpoint)
        return model


class QAModel(torch.nn.Module):
    def __init__(self, config, device):
        super(QAModel, self).__init__()
        self.model = get_hf_model(config).eval()
        self.tokenizer = get_tokenizer(config)
        self.device = device

    def forward(self, batch):
        logits = []
        self.loss = []
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        for item in range(input_ids.size(0)):
            output = self.model(input_ids, attention_mask, labels)
            logits.append(output.logits)
            try:
                self.loss.append(output.loss)
            except:
                pass
        self.loss = torch.stack(self.loss).mean()
        return torch.stack(logits)

    def get_loss(self, logits, batch):
        return self.loss


