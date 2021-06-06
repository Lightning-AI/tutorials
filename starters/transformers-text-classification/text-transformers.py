# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="fqlsVTj7McZ3"
# ### Setup

# %% colab={} colab_type="code" id="OIhHrRL-MnKK"
# !pip install pytorch-lightning datasets transformers

import os
import urllib.request
import zipfile
# %% colab={} colab_type="code" id="6yuQT_ZQMpCg"
from argparse import ArgumentParser
from datetime import datetime
from typing import Optional

import datasets
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, random_split, RandomSampler, TensorDataset
# %% id="vOR0Q1Yg-HmN"
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers.data.processors.glue import MnliProcessor

# %% [markdown] id="7uQVI-xv9Ddj"
# ---
# ## BERT example
# BERT + Lightning

# %% id="e2npX-Gi9uwa"
# ! pip install transformers

# %% [markdown] id="DeLyZQ_E9o1T"
# #### Data download + processing
#
# Let's grab the correct data

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)

# %% id="vMbozzxs9xq_"

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
URL_DATA = "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data"
TASK2PATH = {
    "CoLA": URL_DATA + "%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",
    "SST": URL_DATA + "%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",
    "MRPC": URL_DATA + "%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",
    "QQP": URL_DATA + "%2FQQP-clean.zip?alt=media&token=11a647cb-ecd3-49c9-9d31-79f8ca8fe277",
    "STS": URL_DATA + "%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",
    "MNLI": URL_DATA + "%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",
    "SNLI": URL_DATA + "%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",
    "QNLI": URL_DATA + "%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601",
    "RTE": URL_DATA + "%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",
    "WNLI": URL_DATA + "%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",
    "diagnostic": [
        "https://storage.googleapis.com/mtl-sentence-representations.appspot.com/tsvsWithoutLabels%2FAX.tsv?GoogleAccessId=firebase-adminsdk-0khhl@mtl-sentence-representations.iam.gserviceaccount.com&Expires=2498860800&Signature=DuQ2CSPt2Yfre0C%2BiISrVYrIFaZH1Lc7hBVZDD4ZyR7fZYOMNOUGpi8QxBmTNOrNPjR3z1cggo7WXFfrgECP6FBJSsURv8Ybrue8Ypt%2FTPxbuJ0Xc2FhDi%2BarnecCBFO77RSbfuz%2Bs95hRrYhTnByqu3U%2FYZPaj3tZt5QdfpH2IUROY8LiBXoXS46LE%2FgOQc%2FKN%2BA9SoscRDYsnxHfG0IjXGwHN%2Bf88q6hOmAxeNPx6moDulUF6XMUAaXCSFU%2BnRO2RDL9CapWxj%2BDl7syNyHhB7987hZ80B%2FwFkQ3MEs8auvt5XW1%2Bd4aCU7ytgM69r8JDCwibfhZxpaa4gd50QXQ%3D%3D",  # noqa
        "https://www.dropbox.com/s/ju7d95ifb072q9f/diagnostic-full.tsv?dl=1",
    ],
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def download_and_extract(task, data_dir):
    print("Downloading and extracting %s..." % task)
    data_file = "%s.zip" % task
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    os.remove(data_file)
    print("\tCompleted!")


# %% colab={"base_uri": "https://localhost:8080/", "height": 51} id="3CVHOXQY9yVm" outputId="f06b886b-cc32-4972-918e-f4ca5828fb2c"
download_and_extract('MNLI', '../../notebooks')

processor = MnliProcessor()

# %% [markdown] id="yuUwBKpn-TIK"
# ### Data loaders
#


# %% id="kMdQZUjO-MI7"
def generate_mnli_bert_dataloaders():
    # ----------------------
    # TRAIN/VAL DATALOADERS
    # ----------------------
    train = processor.get_train_examples('MNLI')
    features = convert_examples_to_features(
        train,
        tokenizer,
        label_list=['contradiction', 'neutral', 'entailment'],
        max_length=128,
        output_mode='classification',
        pad_on_left=False,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=0
    )
    train_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.label for f in features], dtype=torch.long)
    )

    nb_train_samples = int(0.95 * len(train_dataset))
    nb_val_samples = len(train_dataset) - nb_train_samples

    bert_mnli_train_dataset, bert_mnli_val_dataset = random_split(train_dataset, [nb_train_samples, nb_val_samples])

    # train loader
    train_sampler = RandomSampler(bert_mnli_train_dataset)
    bert_mnli_train_dataloader = DataLoader(bert_mnli_train_dataset, sampler=train_sampler, batch_size=32)

    # val loader
    val_sampler = RandomSampler(bert_mnli_val_dataset)
    bert_mnli_val_dataloader = DataLoader(bert_mnli_val_dataset, sampler=val_sampler, batch_size=32)

    # ----------------------
    # TEST DATALOADERS
    # ----------------------
    dev = processor.get_dev_examples('MNLI')
    features = convert_examples_to_features(
        dev,
        tokenizer,
        label_list=['contradiction', 'neutral', 'entailment'],
        max_length=128,
        output_mode='classification',
        pad_on_left=False,
        pad_token=tokenizer.pad_token_id,
        pad_token_segment_id=0
    )

    bert_mnli_test_dataset = TensorDataset(
        torch.tensor([f.input_ids for f in features], dtype=torch.long),
        torch.tensor([f.attention_mask for f in features], dtype=torch.long),
        torch.tensor([f.token_type_ids for f in features], dtype=torch.long),
        torch.tensor([f.label for f in features], dtype=torch.long)
    )

    # test dataset
    test_sampler = RandomSampler(bert_mnli_test_dataset)
    bert_mnli_test_dataloader = DataLoader(bert_mnli_test_dataset, sampler=test_sampler, batch_size=32)

    return bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader


# %% id="iV-baDhN-U6B"
bert_mnli_train_dataloader, bert_mnli_val_dataloader, bert_mnli_test_dataloader = generate_mnli_bert_dataloaders()

# %% [markdown] id="yr7eaxkF-djf"
# ### BERT Lightning module!
#
# Finally, we can create the LightningModule

# %% id="UIXLW8CO-W8w"


class BertMNLIFinetuner(pl.LightningModule):

    def __init__(self):
        super(BertMNLIFinetuner, self).__init__()

        self.bert = bert
        self.W = nn.Linear(bert.config.hidden_size, 3)
        self.num_classes = 3

    def forward(self, input_ids, attention_mask, token_type_ids):

        h, _, attn = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        h_cls = h[:, 0]
        logits = self.W(h_cls)
        return logits, attn

    def training_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label)

        # logs
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_nb):
        # batch
        input_ids, attention_mask, token_type_ids, label = batch

        # fwd
        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        # loss
        loss = F.cross_entropy(y_hat, label)

        # acc
        a, y_hat = torch.max(y_hat, dim=1)
        val_acc = accuracy_score(y_hat.cpu(), label.cpu())
        val_acc = torch.tensor(val_acc)

        return {'val_loss': loss, 'val_acc': val_acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        tensorboard_logs = {'val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_loss, 'progress_bar': tensorboard_logs}

    def test_step(self, batch, batch_nb):
        input_ids, attention_mask, token_type_ids, label = batch

        y_hat, attn = self(input_ids, attention_mask, token_type_ids)

        a, y_hat = torch.max(y_hat, dim=1)
        test_acc = accuracy_score(y_hat.cpu(), label.cpu())

        return {'test_acc': torch.tensor(test_acc)}

    def test_epoch_end(self, outputs):

        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()

        tensorboard_logs = {'avg_test_acc': avg_test_acc}
        return {'avg_test_acc': avg_test_acc, 'log': tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def train_dataloader(self):
        return bert_mnli_train_dataloader

    def val_dataloader(self):
        return bert_mnli_val_dataloader

    def test_dataloader(self):
        return bert_mnli_test_dataloader


# %% [markdown] id="FHt8tgwa_DcM"
# ### Trainer

# %% colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["86bedd1fc6da4b8fa0deac637628729e", "f444ab7646444b9885cfec41b5a2236e", "fad0b06dc57e4b4599cf43daad7106b8", "c190999c2761453380f816372fcca608", "a5cc9e60aff641dca27f1adf6807e5b3", "0a96cc26343e4bb2ac2f5145be2fbacf", "cce9ed8de0a048679453e53b71523eea", "773fd1b84c364903bc7350630e76a825", "0e149cc766d147aba2c05f8b0f2c69d5", "191f483b5b0346a8a28cac37f29ac2dc", "24b28a7423a541c0b84ba93d70416c1a", "4820f0005e60493793e506e9f0caf5d4", "fce1fc72006f4e84a6497a493cbbfca2", "f220485e332d4c3cbfc3c45ce3b5fdf1", "bf257b8a04b44a389da2e6f4c64379d4", "7efa007fdb2d4e06b5f34c4286fe9a2f"]} id="gMRMJ-Kd-oup" outputId="790ab73c-b37d-4bcb-af5f-46b464e46f9b"
bert_finetuner = BertMNLIFinetuner()

# most basic trainer, uses good defaults (1 gpu)
trainer = pl.Trainer(gpus=1)
trainer.fit(bert_finetuner)

# %% [markdown] colab_type="text" id="9ORJfiuiNZ_N"
# ## GLUE DataModule


# %% colab={} colab_type="code" id="jW9xQhZxMz1G"
class GLUEDataModule(pl.LightningDataModule):

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

    def setup(self, stage):
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
# #### You could use this datamodule with standalone PyTorch if you wanted...

# %% colab={} colab_type="code" id="JCMH3IAsNffF"
dm = GLUEDataModule('distilbert-base-uncased')
dm.prepare_data()
dm.setup('fit')
next(iter(dm.train_dataloader()))

# %% [markdown] colab_type="text" id="l9fQ_67BO2Lj"
# ## GLUE Model


# %% colab={} colab_type="code" id="gtn5YGKYO65B"
class GLUETransformer(pl.LightningModule):

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
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

    def setup(self, stage):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()

            # Calculate total steps
            tb_size = self.hparams.train_batch_size * max(1, self.hparams.gpus)
            ab_size = self.hparams.accumulate_grad_batches * float(self.hparams.max_epochs)
            self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
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

    @staticmethod
    def add_model_specific_args(parent_parser):
        parent_parser.add_argument_group("GLUETransformer")
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        return parent_parser


# %% [markdown] colab_type="text" id="ha-NdIP_xbd3"
# ### ⚡ Quick Tip
#   - Combine arguments from your DataModule, Model, and Trainer into one for easy and robust configuration


# %% colab={} colab_type="code" id="3dEHnl3RPlAR"
def parse_args(args=None):
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = GLUEDataModule.add_argparse_args(parser)
    parser = GLUETransformer.add_model_specific_args(parser)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args(args)


def main(args):
    pl.seed_everything(args.seed)
    dm = GLUEDataModule.from_argparse_args(args)
    dm.prepare_data()
    dm.setup('fit')
    model = GLUETransformer(num_labels=dm.num_labels, eval_splits=dm.eval_splits, **vars(args))
    trainer = pl.Trainer.from_argparse_args(args)
    return dm, model, trainer


# %% [markdown] colab_type="text" id="PkuLaeec3sJ-"
# # Training

# %% [markdown] colab_type="text" id="QSpueK5UPsN7"
# ## CoLA
#
# See an interactive view of the
# CoLA dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=cola)

# %% colab={} colab_type="code" id="NJnFmtpnPu0Y"
mocked_args = """
    --model_name_or_path albert-base-v2
    --task_name cola
    --max_epochs 3
    --gpus 1""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)

# %% [markdown] colab_type="text" id="_MrNsTnqdz4z"
# ## MRPC
#
# See an interactive view of the
# MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mrpc)

# %% colab={} colab_type="code" id="LBwRxg9Cb3d-"
mocked_args = """
    --model_name_or_path distilbert-base-cased
    --task_name mrpc
    --max_epochs 3
    --gpus 1""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)

# %% [markdown] colab_type="text" id="iZhbn0HzfdCu"
# ## MNLI
#
#  - The MNLI dataset is huge, so we aren't going to bother trying to train it here.
#  - Let's just make sure our multi-dataloader logic is right by skipping over training and going straight to validation.
#
# See an interactive view of the
# MRPC dataset in [NLP Viewer](https://huggingface.co/nlp/viewer/?dataset=glue&config=mnli)

# %% colab={} colab_type="code" id="AvsZMOggfcWW"
mocked_args = """
    --model_name_or_path distilbert-base-uncased
    --task_name mnli
    --max_epochs 1
    --gpus 1
    --limit_train_batches 10
    --progress_bar_refresh_rate 20""".split()

args = parse_args(mocked_args)
dm, model, trainer = main(args)
trainer.fit(model, dm)
