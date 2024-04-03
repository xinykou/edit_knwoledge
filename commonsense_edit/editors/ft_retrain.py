import torch
from commonsense_edit.utils import param_subset, brackets_to_periods
from transformers import get_linear_schedule_with_warmup

class Finetune_retrain(torch.nn.Module):
    def __init__(self, config, model):
        """
        This method directly finetunes chosen weights given new inputs
        """
        super(Finetune_retrain, self).__init__()
        self.model = model.model
        self.tokenizer = model.tokenizer
        self.pnames = brackets_to_periods(config.editor.inner_params[0])
        self.edit_lr = config.model.fine_tune.learning_rate
        self.retrain_memory = config.editor.retrain_memory

        self.config = config

        for n, p in self.model.named_parameters():
            if n != self.pnames:
                p.requires_grad = False
            else:
                p.requires_grad = True

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids, attention_mask, labels)

    def retrain(self, batch_history):
        params = param_subset(self.model.named_parameters(), [self.pnames])
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        scheduler = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.config.editor.n_iter,
        )
        for _ in range(5):
            for tokens in batch_history:
                self.model.zero_grad()
                outputs = self.model(**tokens)
                logits, loss = outputs.logits, outputs.loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                scheduler.step()
        
    def edit(self, config=None, tokens=None, batch_history=None):
        params = param_subset(self.model.named_parameters(), [self.pnames])
        opt = torch.optim.Adam(params, lr=self.edit_lr)
        scheduler = get_linear_schedule_with_warmup(
            opt,
            num_warmup_steps=0,
            num_training_steps=self.config.editor.n_iter,
        )
        self.losses = []
        for _ in range(self.config.editor.n_iter):
            self.model.zero_grad()
            outputs = self.model(**tokens)
            logits, loss1 = outputs.logits, outputs.loss
            argmaxs = torch.argmax(logits, dim=-1)
            response_indices = (tokens['labels'] != -100)
            if torch.all(tokens['labels'][response_indices] == argmaxs[response_indices]).item():
                break
            self.loss = loss1
            self.losses.append(self.loss.detach().cpu().numpy())
            opt.zero_grad()
            self.loss.backward()
            opt.step()
            scheduler.step()
        return self.model
