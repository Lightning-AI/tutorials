# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% colab={} colab_type="code" id="6yuQT_ZQMpCg"
from datetime import datetime
from typing import Optional

import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything, Trainer
from torch.utils.data import DataLoader
# %% id="vOR0Q1Yg-HmN"
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# %% [markdown] id="7uQVI-xv9Ddj"
# ---
# ## Training BERT with Lightning

# %% [markdown] colab_type="text" id="9ORJfiuiNZ_N"
# ### Lightning DataModule for GLUE


# %% colab={} colab_type="code" id="jW9xQhZxMz1G"
class GLUEDataModule(LightningDataModule):

    task_text_field_map = {
        'cola': ['sentence'],
        'sst2': ['sentence'],
        'mrpc': ['sentence1', 'sentence2'],
        'qqp': ['question1', 'question2'],
        'stsb': ['sentence1', 'sentence2'],
        'mnli': ['premise', 'hypothesis'],
        'qnli': ['question', 'sentence'],
        'rte': ['sentence1', 'sentence2'],
        'wnli': ['sentence1', 'sentence2'],
        'ax': ['premise', 'hypothesis']
    }

    glue_task_num_labels = {
        'cola': 2,
        'sst2': 2,
        'mrpc': 2,
        'qqp': 2,
        'stsb': 1,
        'mnli': 3,
        'qnli': 2,
        'rte': 2,
        'wnli': 2,
        'ax': 3
    }

    loader_columns = [
        'datasets_idx', 'input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions', 'labels'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = 'mrpc',
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset('glue', self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=['label'],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        self.eval_splits = [x for x in self.dataset.keys() if 'validation' in x]

    def prepare_data(self):
        datasets.load_dataset('glue', self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset['train'], batch_size=self.train_batch_size)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['validation'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset['test'], batch_size=self.eval_batch_size)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):

        # Either encode single sentence or sentence pairs
        if len(self.text_fields) > 1:
            texts_or_text_pairs = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))
        else:
            texts_or_text_pairs = example_batch[self.text_fields[0]]

        # Tokenize the text/text pairs
        features = self.tokenizer.batch_encode_plus(
            texts_or_text_pairs, max_length=self.max_seq_length, pad_to_max_length=True, truncation=True
        )

        # Rename label to labels to make it easier to pass to model forward
        features['labels'] = example_batch['label']

        return features


# %% [markdown] colab_type="text" id="jQC3a6KuOpX3"
# **You could use this datamodule with standalone PyTorch if you wanted...**

# %% colab={} colab_type="code" id="JCMH3IAsNffF"
dm = GLUEDataModule('distilbert-base-uncased')
dm.prepare_data()
dm.setup('fit')
next(iter(dm.train_dataloader()))

# %% [markdown] colab_type="text" id="l9fQ_67BO2Lj"
# ### Transformer LightningModule


# %% colab={} colab_type="code" id="gtn5YGKYO65B"
class GLUETransformer(LightningModule):

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str,
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metric = datasets.load_metric(
            'glue', self.hparams.task_name, experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == 'mnli':
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split('_')[-1]
                preds = torch.cat([x['preds'] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x['labels'] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x['loss'] for x in output]).mean()
                self.log(f'val_loss_{split}', loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v
                    for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
            ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return [optimizer], [scheduler]


# %% [markdown] colab_type="text" id="QSpueK5UPsN7"
# ## Training

# %% [markdown] colab_type="text" id="QSpueK5UPsN7"
# #### CoLA
#
# See an interactive view of the
# CoLA dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=cola)

# %% colab={} colab_type="code" id="NJnFmtpnPu0Y"
seed_everything(42)

dm = GLUEDataModule(model_name_or_path='albert-base-v2', task_name='cola')
dm.setup('fit')
model = GLUETransformer(
    model_name_or_path='albert-base-v2', num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name
)

trainer = Trainer(max_epochs=3, gpus=min(1, torch.cuda.device_count()))
trainer.fit(model, dm)

# %% [markdown] colab_type="text" id="_MrNsTnqdz4z"
# #### MRPC
#
# See an interactive view of the
# MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mrpc)

# %% colab={} colab_type="code" id="LBwRxg9Cb3d-"
seed_everything(42)

dm = GLUEDataModule(model_name_or_path='distilbert-base-cased', task_name='mrpc')
dm.setup('fit')
model = GLUETransformer(
    model_name_or_path='distilbert-base-cased',
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name
)

trainer = Trainer(max_epochs=3, gpus=min(1, torch.cuda.device_count()))
trainer.fit(model, dm)

# %% [markdown] colab_type="text" id="iZhbn0HzfdCu"
# #### MNLI
#
#  - The MNLI dataset is huge, so we aren't going to bother trying to train on it here.
#  - We will skip over training and go straight to validation.
#
# See an interactive view of the
# MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mnli)

# %% colab={} colab_type="code" id="AvsZMOggfcWW"
dm = GLUEDataModule(model_name_or_path='distilbert-base-cased', task_name='mnli')
dm.setup('fit')
model = GLUETransformer(
    model_name_or_path='distilbert-base-cased',
    num_labels=dm.num_labels,
    eval_splits=dm.eval_splits,
    task_name=dm.task_name
)

trainer = Trainer(gpus=min(1, torch.cuda.device_count()), progress_bar_refresh_rate=20)
trainer.validate(model, dm.val_dataloader())
