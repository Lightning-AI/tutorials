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
