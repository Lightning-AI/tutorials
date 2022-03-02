# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Fast Multi-agent Reinforcement Learning on a GPU using WarpDrive and Pytorch Lightning

# %% [markdown]
# # ⚠️ PLEASE NOTE:
# This notebook runs on a GPU runtime.\
# If running on Colab, choose Runtime > Change runtime type from the menu, then select `GPU` in the 'Hardware accelerator' dropdown menu.

# %% [markdown]
# # Introduction

# %% [markdown]
# This tutorial provides a demonstration of a multi-agent Reinforcement Learning (RL) training loop with [WarpDrive](https://github.com/salesforce/warp-drive). WarpDrive is a flexible, lightweight, and easy-to-use RL framework that implements end-to-end deep multi-agent RL on a single GPU (Graphics Processing Unit). Using the extreme parallelization capability of GPUs, it enables [orders-of-magnitude faster RL](https://arxiv.org/abs/2108.13976) compared to common implementations that blend CPU simulations and GPU models. WarpDrive is extremely efficient as it runs simulations across multiple agents and multiple environment replicas in parallel and completely eliminates the back-and-forth data copying between the CPU and the GPU.
#
# We have integrated WarpDrive with the [Pytorch Lightning](https://www.pytorchlightning.ai/) framework, which greatly reduces the trainer boilerplate code, and improves training flexibility.
#
# Below, we demonstrate how to use WarpDrive and PytorchLightning together to train a game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) where multiple *tagger* agents are trying to run after and tag multiple other *runner* agents. As such, the Warpdrive framework comprises several utility functions that help easily implement any (OpenAI-)*gym-style* RL environment, and furthermore, provides quality-of-life tools to train it end-to-end using just a few lines of code. You may familiarize yourself with WarpDrive with the help of these [tutorials](https://github.com/salesforce/warp-drive/tree/master/tutorials).
#
# We invite everyone to **contribute to WarpDrive**, including adding new multi-agent environments, proposing new features and reporting issues on our open source [repository](https://github.com/salesforce/warp-drive).

# %% [markdown]
# ### Dependencies

# %% [markdown]
# This notebook requires the `rl-warp-drive` as well as the `pytorch-lightning` packages.

# %%
# ! pip install --quiet 'rl_warp_drive>=1.5' 'pytorch_lightning>=1.5.10'

# %%
import torch

_NUM_AVAILABLE_GPUS = torch.cuda.device_count()
assert _NUM_AVAILABLE_GPUS > 0, "This notebook needs a GPU to run!"

# %%
# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
import logging

from example_envs.tag_continuous.tag_continuous import TagContinuous
from pytorch_lightning import Trainer
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.lightning_trainer import PerfStatsCallback, WarpDriveModule
from warp_drive.training.utils.data_loader import create_and_push_data_placeholders

logging.getLogger().setLevel(logging.ERROR)

# %% [markdown]
# # Specify a set of run configurations for your experiments

# %% [markdown]
# The run configuration is a dictionary comprising the environment parameters, the trainer and the policy network settings, as well as configurations for saving.
#
# For our experiment, we consider an environment wherein $5$ taggers and $100$ runners play the game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) on a $20 \times 20$ plane. The game lasts $200$ timesteps. Each agent chooses it's own acceleration and turn actions at every timestep, and we use mechanics to determine how the agents move over the grid. When a tagger gets close to a runner, the runner is tagged, and is eliminated from the game. For the configuration below, the runners and taggers have the same skill levels, or top speeds. The animation below shows a sample realization of the game episode before training, i.e., with randomly chosen agent actions. The $5$ taggers are marked in pink, while the $100$ blue agents are the runners. Both the taggers and runners move around randomly and over half the runners remain at the end of the episode.
#
# <img src="assets/tag_before_training.gif" width="500" height="500"/>

# %% [markdown]
# We train the agents using $50$ environments or simulations running in parallel. With WarpDrive, each simulation runs on sepate GPU blocks.
#
# There are two separate policy networks used for the tagger and runner agents. Each network is a fully-connected model with two layers each of $256$ dimensions. We use the Advantage Actor Critic (A2C) algorithm for training. WarpDrive also currently provides the option to use the Proximal Policy Optimization (PPO) algorithm instead.

# %%
run_config = dict(
    name="tag_continuous",
    # Environment settings.
    env=dict(
        num_taggers=5,  # number of taggers in the environment
        num_runners=100,  # number of runners in the environment
        grid_length=20.0,  # length of the (square) grid on which the game is played
        episode_length=200,  # episode length in timesteps
        max_acceleration=0.1,  # maximum acceleration
        min_acceleration=-0.1,  # minimum acceleration
        max_turn=2.35,  # 3*pi/4 radians
        min_turn=-2.35,  # -3*pi/4 radians
        num_acceleration_levels=10,  # number of discretized accelerate actions
        num_turn_levels=10,  # number of discretized turn actions
        skill_level_tagger=1.0,  # skill level for the tagger
        skill_level_runner=1.0,  # skill level for the runner
        use_full_observation=False,  # each agent only sees full or partial information
        runner_exits_game_after_tagged=True,  # flag to indicate if a runner stays in the game after getting tagged
        num_other_agents_observed=10,  # number of other agents each agent can see
        tag_reward_for_tagger=10.0,  # positive reward for the tagger upon tagging a runner
        tag_penalty_for_runner=-10.0,  # negative reward for the runner upon getting tagged
        end_of_game_reward_for_runner=1.0,  # reward at the end of the game for a runner that isn't tagged
        tagging_distance=0.02,  # margin between a tagger and runner to consider the runner as 'tagged'.
    ),
    # Trainer settings.
    trainer=dict(
        num_envs=50,  # number of environment replicas (number of GPU blocks used)
        train_batch_size=10000,  # total batch size used for training per iteration (across all the environments)
        num_episodes=10000,  # total number of episodes to run the training for (can be arbitrarily high!)
    ),
    # Policy network settings.
    policy=dict(
        runner=dict(
            to_train=True,  # flag indicating whether the model needs to be trained
            algorithm="A2C",  # algorithm used to train the policy
            gamma=0.98,  # discount rate
            lr=0.005,  # learning rate
            model=dict(
                type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""
            ),  # policy model settings
        ),
        tagger=dict(
            to_train=True,
            algorithm="A2C",
            gamma=0.98,
            lr=0.002,
            model=dict(
                type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""
            ),
        ),
    ),
    # Checkpoint saving setting.
    saving=dict(
        metrics_log_freq=10,  # how often (in iterations) to print the metrics
        model_params_save_freq=5000,  # how often (in iterations) to save the model parameters
        basedir="/tmp",  # base folder used for saving
        name="continuous_tag",  # experiment name
        tag="example",  # experiment tag
    ),
)

# %% [markdown]
# # Instantiate the WarpDrive Module

# %% [markdown]
# In order to instantiate the WarpDrive module, we first use an environment wrapper to specify that the environment needs to be run on the GPU (via the `use_cuda` flag). Also, agents in the environment can share policy models; so we specify a dictionary to map each policy network model to the list of agent ids using that model.

# %%
# Create a wrapped environment object via the EnvWrapper.
# Ensure that use_cuda is set to True (in order to run on the GPU).
env_wrapper = EnvWrapper(
    TagContinuous(**run_config["env"]),
    num_envs=run_config["trainer"]["num_envs"],
    use_cuda=True,
)

# Agents can share policy models: this dictionary maps policy model names to agent ids.
policy_tag_to_agent_id_map = {
    "tagger": list(env_wrapper.env.taggers),
    "runner": list(env_wrapper.env.runners),
}

wd_module = WarpDriveModule(
    env_wrapper=env_wrapper,
    config=run_config,
    policy_tag_to_agent_id_map=policy_tag_to_agent_id_map,
    verbose=True,
)

# %% [markdown]
# # Create the Lightning Trainer

# %% [markdown]
# Next, we create the trainer for training the WarpDrive model. We add the `performance stats` callbacks to the trainer to view the throughput performance of WarpDrive.

# %%
log_freq = run_config["saving"]["metrics_log_freq"]

# Define callbacks.
perf_stats_callback = PerfStatsCallback(
    batch_size=wd_module.training_batch_size,
    num_iters=wd_module.num_iters,
    log_freq=log_freq,
)

# Instantiate the PytorchLightning trainer with the callbacks and the number of gpus.
num_gpus = 1
assert (
    num_gpus <= _NUM_AVAILABLE_GPUS
), f"Only {_NUM_AVAILABLE_GPUS} GPU(s) are available!"
num_epochs = run_config["trainer"]["num_episodes"] * \
             run_config["env"]["episode_length"] / \
             run_config["trainer"]["train_batch_size"]

trainer = Trainer(
    gpus=num_gpus,
    callbacks=[perf_stats_callback],
    max_epochs=num_epochs,
)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# %% [markdown]
# # Train the WarpDrive Module

# %% [markdown]
# Finally, we invoke training.
#
# Note: please scroll up to the tensorboard cell to visualize the curves during training.

# %%
trainer.fit(wd_module)

# %% [markdown]
# ## Visualize an episode-rollout after training (for 2M episodes)

# %% [markdown]
# In the configuration above, we have set the trainer to only train on $10000$ rollout episodes, but we also trained it offline for $2$M episodes (or $400$M steps), which takes about 6 hours of training on an A100 GPU. Over training, the runners learn to escape the taggers, and the taggers learn to chase after the runner. Sometimes, the taggers also collaborate to team-tag runners.
# The animation below shows a sample realization of the game episode after training for $2$M episodes.
#
# <img src="assets/tag_after_training.gif" width="500" height="500"/>

# %% [markdown]
# # Learn More about WarpDrive and explore our tutorials!

# %% [markdown]
# 1. [WarpDrive basics](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-1-warp_drive_basics.ipynb)
# 2. [WarpDrive sampler](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-2-warp_drive_sampler.ipynb)
# 3. [WarpDrive reset and log](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-3-warp_drive_reset_and_log.ipynb)
# 4. [Creating custom environments](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-4-create_custom_environments.md)
# 5. [Training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-5-training_with_warp_drive.ipynb)
# 6. [Scaling Up training with WarpDrive](https://www.github.com/salesforce/warp-drive/blob/master/tutorials/tutorial-6-scaling_up_training_with_warp_drive.md)
