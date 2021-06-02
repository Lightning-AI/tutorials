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

# %% [markdown] id="jKj5lgdr5j48"
# ---
# ### Setup
# Lightning is easy to use. Simply ```pip install pytorch-lightning```

# %% colab={"base_uri": "https://localhost:8080/", "height": 938} id="UGjilEHk4vb7" outputId="229670cf-ec26-446f-afe5-2432c4571030"
# ! pip install pytorch-lightning --upgrade

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

# %% colab={"base_uri": "https://localhost:8080/", "height": 164, "referenced_widgets": ["5484eef7b6f247d68a89f86965b0940f", "0c3473a16a5e4c46a6c7515e610bca7f", "ad849800b2124195b92f3bf9dfc7681b", "6ae5b2f9195847b5a0aa9991e14aa397", "240764252e7c4f5ca39db14fd1c724ed", "386ff59e3694480394253f1c24ff8e84", "70e48d7d8e8a411a90642926db4aada8", "1f3364ab59b541268fabcb3f9fb5c64c", "0fad6468e3c849b380e34f674e074219", "10a88a05740b45d4a6ea5873d4a7151a", "d3b107acd1b1401cabe3090724e12e86", "b3563100dd1b4a4abe14ab7193649064", "17f0e360e85f48d9a17b84c9b7f6c9f0", "29f35103a6e94af09c8ac9cdb2cca89c", "e6e15d5c14134be0b4cf86fdecfef687", "f23f02d00d424574afa29311b8d0906e", "e918a6de59b64bd590e4f1233bbc078a", "abeb0a773f3542c39ff724ae0674b74e", "892246fdf6bb476abb35ec321ddf86e8", "88c181cd21a94ec9a43df9754c1986c9", "e4098b0091124fef8ba342783a82cc6e", "498a50387a0742a88356a7ee9920bf7a", "86482894cddd4956ae2fc3d9edd8ef9a", "438d19fb8e8243ebbc658f4b1d27df99"]} id="eBP6FeY18_Ck" outputId="b2a5c5fd-88cf-4428-d196-9e1c1ddc7e30"
from transformers.data.processors.glue import MnliProcessor
from transformers import (BertModel, BertTokenizer)

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
bert = BertModel.from_pretrained('bert-base-cased', output_attentions=True)

# %% id="vMbozzxs9xq_"
import os
import urllib.request
import zipfile

TASKS = ["CoLA", "SST", "MRPC", "QQP", "STS", "MNLI", "SNLI", "QNLI", "RTE", "WNLI", "diagnostic"]
TASK2PATH = {
    "CoLA": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FCoLA.zip?alt=media&token=46d5e637-3411-4188-bc44-5809b5bfb5f4",  # noqa
    "SST": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSST-2.zip?alt=media&token=aabc5f6b-e466-44a2-b9b4-cf6337f84ac8",  # noqa
    "MRPC": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2Fmrpc_dev_ids.tsv?alt=media&token=ec5c0836-31d5-48f4-b431-7480817f1adc",  # noqa
    "QQP": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQQP-clean.zip?alt=media&token=11a647cb-ecd3-49c9-9d31-79f8ca8fe277",  # noqa
    "STS": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSTS-B.zip?alt=media&token=bddb94a7-8706-4e0d-a694-1109e12273b5",  # noqa
    "MNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FMNLI.zip?alt=media&token=50329ea1-e339-40e2-809c-10c40afff3ce",  # noqa
    "SNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FSNLI.zip?alt=media&token=4afcfbb2-ff0c-4b2d-a09a-dbf07926f4df",  # noqa
    "QNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLIv2.zip?alt=media&token=6fdcf570-0fc5-4631-8456-9505272d1601",  # noqa
    "RTE": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FRTE.zip?alt=media&token=5efa7e85-a0bb-4f19-8ea2-9e1840f077fb",  # noqa
    "WNLI": "https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FWNLI.zip?alt=media&token=068ad0a0-ded7-4bd7-99a5-5e00222e0faf",  # noqa
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

# %% id="vOR0Q1Yg-HmN"
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from torch.utils.data import TensorDataset, RandomSampler, random_split

processor = MnliProcessor()

# %% [markdown] id="yuUwBKpn-TIK"
# #### Data loaders
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
from sklearn.metrics import accuracy_score
import torch.nn.functional as F


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

# %% [markdown] id="NWvMLBDySQI5"
# ## DQN example
#
# How to train a Deep Q Network
#
# Main takeaways:
# 1. RL has the same flow as previous models we have seen, with a few additions
# 2. Handle unsupervised learning by using an IterableDataset where the dataset itself is constantly updated during training
# 3. Each training step carries has the agent taking an action in the environment and storing the experience in the IterableDataset

# %% colab={"base_uri": "https://localhost:8080/", "height": 146} id="4ARIT37rDdIZ" outputId="37ea5092-0db7-4e73-b507-f4be9bb0ae7e"
# !pip install gym

# %% [markdown] id="nm9BKoF0Sv_O"
# ### DQN Network

# %% id="FXkKtnEhSaIV"
from torch import nn


class DQN(nn.Module):
    """
    Simple MLP network

    Args:
        obs_size: observation/state size of the environment
        n_actions: number of discrete actions available in the environment
        hidden_size: size of hidden layers
    """

    def __init__(self, obs_size: int, n_actions: int, hidden_size: int = 128):
        super(DQN, self).__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.net(x.float())


# %% [markdown] id="c9clSz7xTFZf"
# ### Memory

# %% id="zUmawp0ITE3I"
from collections import namedtuple

# Named tuple for storing experience steps gathered in training
Experience = namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

# %% id="Zs7h_Z0LTVoy"
from typing import Tuple


class ReplayBuffer:
    """
    Replay Buffer for storing past experiences allowing the agent to learn from them

    Args:
        capacity: size of the buffer
    """

    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def __len__(self) -> None:
        return len(self.buffer)

    def append(self, experience: Experience) -> None:
        """
        Add experience to the buffer

        Args:
            experience: tuple (state, action, reward, done, new_state)
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int) -> Tuple:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])

        return (
            np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(dones, dtype=np.bool),
            np.array(next_states)
        )


# %% id="R5UK2VRvTgS1"
from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """
    Iterable Dataset containing the ExperienceBuffer
    which will be updated with new experiences during training

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: ReplayBuffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Tuple:
        states, actions, rewards, dones, new_states = self.buffer.sample(self.sample_size)
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]


# %% [markdown] id="d7sCGSURTuQK"
# ### Agent

# %% id="dS2RpSHHTvpO"
import gym
import torch


class Agent:
    """
    Base Agent class handeling the interaction with the environment

    Args:
        env: training environment
        replay_buffer: replay buffer storing experiences
    """

    def __init__(self, env: gym.Env, replay_buffer: ReplayBuffer) -> None:
        self.env = env
        self.replay_buffer = replay_buffer
        self.reset()
        self.state = self.env.reset()

    def reset(self) -> None:
        """ Resents the environment and updates the state"""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, epsilon: float, device: str) -> int:
        """
        Using the given network, decide what action to carry out
        using an epsilon-greedy policy

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            action
        """
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor([self.state])

            if device not in ['cpu']:
                state = state.cuda(device)

            q_values = net(state)
            _, action = torch.max(q_values, dim=1)
            action = int(action.item())

        return action

    @torch.no_grad()
    def play_step(self, net: nn.Module, epsilon: float = 0.0, device: str = 'cpu') -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment

        Args:
            net: DQN network
            epsilon: value to determine likelihood of taking a random action
            device: current device

        Returns:
            reward, done
        """

        action = self.get_action(net, epsilon, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)

        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done


# %% [markdown] id="IAlT0-75T_Kv"
# ### DQN Lightning Module

# %% id="BS5D7s83T13H"
import pytorch_lightning as pl
import argparse
from collections import OrderedDict, deque
from typing import Tuple, List
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()
        self.hparams = hparams

        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.net = DQN(obs_size, n_actions)
        self.target_net = DQN(obs_size, n_actions)

        self.buffer = ReplayBuffer(self.hparams.replay_size)
        self.agent = Agent(self.env, self.buffer)
        self.total_reward = 0
        self.episode_reward = 0
        self.populate(self.hparams.warm_start_steps)

    def populate(self, steps: int = 1000) -> None:
        """
        Carries out several random steps through the environment to initially fill
        up the replay buffer with experiences

        Args:
            steps: number of random steps to populate the buffer with
        """
        for i in range(steps):
            self.agent.play_step(self.net, epsilon=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def dqn_mse_loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.hparams.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], nb_batch) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            nb_batch: batch number

        Returns:
            Training loss and log metrics
        """
        device = self.get_device(batch)
        epsilon = max(self.hparams.eps_end, self.hparams.eps_start - self.global_step + 1 / self.hparams.eps_last_frame)

        # step through environment with agent
        reward, done = self.agent.play_step(self.net, epsilon, device)
        self.episode_reward += reward

        # calculates training loss
        loss = self.dqn_mse_loss(batch)

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss = loss.unsqueeze(0)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # Soft update of target network
        if self.global_step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        log = {
            'total_reward': torch.tensor(self.total_reward).to(device),
            'reward': torch.tensor(reward).to(device),
            'train_loss': loss
        }
        status = {
            'steps': torch.tensor(self.global_step).to(device),
            'total_reward': torch.tensor(self.total_reward).to(device)
        }

        return OrderedDict({'loss': loss, 'log': log, 'progress_bar': status})

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.hparams.lr)
        return [optimizer]

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch"""
        return batch[0].device.index if self.on_gpu else 'cpu'


# %% [markdown] id="JST5AN-8VFLY"
# ### Trainer


# %% id="bQEvD7gFUSaN"
def main(hparams) -> None:
    model = DQNLightning(hparams)

    trainer = pl.Trainer(
        gpus=1, distributed_backend='dp', max_epochs=500, early_stop_callback=False, val_check_interval=100
    )

    trainer.fit(model)


# %% colab={"base_uri": "https://localhost:8080/", "height": 380, "referenced_widgets": ["e9a6bf4eda3244c6bb17216715f36525", "0922c5b2de554b4fa28dd531603f2709", "c293fc4171b0438595bc9a49fbb250cf", "819c83bf0bbd472ba417c31e957718c7", "c24384195a074989a86217b2edc411cb", "b3817e0ba30f449585f7641b4d3061bb", "8591bd2136ab4bb7831579609b43ee9c", "5a761ed145474ec7a30006bc584b26be"]} id="-iV9PQC9VOHK" outputId="2fd70097-c913-4d68-e80a-d240532edd19"
import numpy as np
import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
parser.add_argument("--env", type=str, default="CartPole-v0", help="gym environment tag")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
parser.add_argument("--sync_rate", type=int, default=10, help="how many frames do we update the target network")
parser.add_argument("--replay_size", type=int, default=1000, help="capacity of the replay buffer")
parser.add_argument(
    "--warm_start_size",
    type=int,
    default=1000,
    help="how many samples do we use to fill our buffer at the start of training"
)
parser.add_argument("--eps_last_frame", type=int, default=1000, help="what frame should epsilon stop decaying")
parser.add_argument("--eps_start", type=float, default=1.0, help="starting value of epsilon")
parser.add_argument("--eps_end", type=float, default=0.01, help="final value of epsilon")
parser.add_argument("--episode_length", type=int, default=200, help="max length of an episode")
parser.add_argument("--max_episode_reward", type=int, default=200, help="max episode reward in the environment")
parser.add_argument("--warm_start_steps", type=int, default=1000, help="max episode reward in the environment")

args, _ = parser.parse_known_args()

main(args)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
