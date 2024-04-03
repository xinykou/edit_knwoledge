import os.path

import hydra
from omegaconf import DictConfig, OmegaConf
import omegaconf
import logging
from editors import Finetune, Lora_Efficient, Lora_Postfusion, Lora_Postfusion_Expert, Lora_Postfusion_Vip, Lora_Postfusion_Layers, Prefix_Postfusion_Layers, Lora_Postfusion_Bias   # todo：这里是引用的模块
from dataset.dataset import DisentQADataset, Contextual_Question_Separated_Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.utilities.seed import seed_everything
import transformers
logging.basicConfig(format='%(asctime)s - %(levelname)s [%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)
LOG = logging.getLogger(__name__)

@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    LOG.info(f"\n\n{OmegaConf.to_yaml(config)}\n")
    train_source = config.train_type
    editor_type = config.editor_type
    train_path = config.experiment[train_source]
    train_config = config.model.fine_tune

    seed_everything(42)

    Tokenzir_Class = getattr(transformers, config.model.tokenizer_class)
    tokenizer = Tokenzir_Class.from_pretrained(config.model.model_cache)
    using_fn = False

    if config.editor._name == "ft":
        # model = Finetune(config)
        ModelClass = globals()["Finetune"]
        train_dataset = DisentQADataset(config, tokenizer, train_path,
                                        source_max_token_len=train_config.source_max_token_len,
                                        target_max_token_len=train_config.target_max_token_len)

    elif config.editor._name == "lora_ef" or config.editor._name == "lora_hyperdecoders":
        # model = Lora_Efficient(config)
        ModelClass = globals()["Lora_Efficient"]
        train_dataset = DisentQADataset(config, tokenizer, train_path,
                                        source_max_token_len=train_config.source_max_token_len,
                                        target_max_token_len=train_config.target_max_token_len)


    elif config.editor._name == "lora_hyperdecoders_postfusion":
        ModelClass = globals()['Lora_Postfusion']
        train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                        source_max_token_len=train_config.source_max_token_len,
                                        target_max_token_len=train_config.target_max_token_len,
                                        context_max_token_len=train_config.context_max_token_len)
        using_fn = True
        # # debug
        # train_dataloader = DataLoader(train_dataset, batch_size=train_config.batch_size, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
        # for b in train_dataloader:
        #     print()

    elif config.editor._name == "lora_hyperdecoders_postfusion_mixexperts":
        ModelClass = globals()['Lora_Postfusion_Expert']
        train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                                              source_max_token_len=train_config.source_max_token_len,
                                                              target_max_token_len=train_config.target_max_token_len,
                                                              context_max_token_len=train_config.context_max_token_len)
        using_fn = True

    elif config.editor._name == "lora_hyperdecoders_postfusion_vip":
        ModelClass = globals()['Lora_Postfusion_Vip']
        train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                                              source_max_token_len=train_config.source_max_token_len,
                                                              target_max_token_len=train_config.target_max_token_len,
                                                              context_max_token_len=train_config.context_max_token_len)
        using_fn = True

    elif config.editor._name == "lora_hyperdecoders_postfusion_layers":
        ModelClass = globals()['Lora_Postfusion_Layers']
        train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                        source_max_token_len=train_config.source_max_token_len,
                                        target_max_token_len=train_config.target_max_token_len,
                                        context_max_token_len=train_config.context_max_token_len)
        using_fn = True
    elif config.editor._name == "prefix_hyperdecoders_postfusion_layers":
         ModelClass = globals()['Prefix_Postfusion_Layers']
         train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                                          source_max_token_len=train_config.source_max_token_len,
                                                          target_max_token_len=train_config.target_max_token_len,
                                                          context_max_token_len=train_config.context_max_token_len)
         using_fn = True

    elif config.editor._name == "lora_hyperdecoders_postfusion_bias":
         ModelClass = globals()['Lora_Postfusion_Bias']
         train_dataset = Contextual_Question_Separated_Dataset(config, tokenizer, train_path,
                                                          source_max_token_len=train_config.source_max_token_len,
                                                          target_max_token_len=train_config.target_max_token_len,
                                                          context_max_token_len=train_config.context_max_token_len)
         using_fn = True

    if config.train_mode == "train":

        checkpoint_filename = config.checkpoint_filename
        checkpoint_path = os.path.join(train_config.checkpoints_dirpath, checkpoint_filename)

        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print("New Parameter name %s", name)

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=train_config.batch_size,
                                      shuffle=True,
                                      num_workers=4,
                                      collate_fn=train_dataset.collate_fn if using_fn else None)

        val_dataloader = DataLoader(train_dataset,
                                    batch_size=train_config.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=train_dataset.collate_fn if using_fn else None)

        new_config = omegaconf.DictConfig({'data_size': len(train_dataloader)})  # 样本集可以分为多少个批次
        config = {**config, **new_config}
        config = omegaconf.DictConfig(config)
        print(config)

        model = ModelClass.load_from_checkpoint(checkpoint_path, strict=False, config=config)

    checkpoint_filename = editor_type + "-" + train_source + "_" + config.model.model_name
    try:
        random_evidence = config.random_evidence
        noisy_evidence = config.noisy_evidence
    except:
        random_evidence = False
        noisy_evidence = False
    if random_evidence:
        checkpoint_filename = 'random' + checkpoint_filename
    elif noisy_evidence:
        checkpoint_filename = 'noisy' + checkpoint_filename
    checkpoint_callback = ModelCheckpoint(dirpath=train_config.checkpoints_dirpath,
                                          filename=checkpoint_filename + '-{epoch:02d}-{val_loss:.4f}',
                                          save_top_k=3,
                                          verbose=True, monitor="val_loss", mode="min")

    # trainer = pl.Trainer(callbacks=checkpoint_callback, max_epochs=train_config.n_epochs, devices=[0, 1], accelerator="gpu",
    #                     strategy="ddp")
    trainer = pl.Trainer(callbacks=checkpoint_callback, max_epochs=train_config.n_epochs, gpus=[0])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == "__main__":
    run()