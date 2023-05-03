# %% [markdown]
# **⚠️ PLEASE NOTE:**
# This notebook runs on a GPU runtime. If running on Colab, choose Runtime > Change runtime type from the menu, then select `GPU` in the 'Hardware accelerator' dropdown menu.

# %% [markdown]
# ## Introduction

# %% [markdown]
# This tutorial provides a demonstration of a multi-agent Reinforcement Learning (RL) training loop with [WarpDrive](https://github.com/salesforce/warp-drive). WarpDrive is a flexible, lightweight, and easy-to-use RL framework that implements end-to-end deep multi-agent RL on a GPU (Graphics Processing Unit). Using the extreme parallelization capability of GPUs, it enables [orders-of-magnitude faster RL](https://arxiv.org/abs/2108.13976) compared to common implementations that blend CPU simulations and GPU models. WarpDrive is extremely efficient as it runs simulations across multiple agents and multiple environment replicas all in parallel and completely eliminates the back-and-forth data copying between the CPU and the GPU during every step. As such, WarpDrive
# - Can simulate 1000s of agents in each environment and thousands of environments in parallel, harnessing the extreme parallelism capability of GPUs.
# - Eliminates communication between CPU and GPU, and also within the GPU, as read and write operations occur in-place.
# - Is fully compatible with PyTorch, a highly flexible and very fast deep learning framework.
# - Implements parallel action sampling on CUDA C, which is ~3x faster than using PyTorch’s sampling methods.
# - Allows for large-scale distributed training on multiple GPUs.
#
# Below is an overview of WarpDrive’s layout of computational and data structures on a single GPU.
# ![](https://blog.salesforceairesearch.com/content/images/2021/08/warpdrive_framework_overview.png)
# Computations are organized into blocks, with multiple threads in each block. Each block runs a simulation environment and each thread
# simulates an agent in an environment. Blocks can access the shared GPU memory that stores simulation data and neural network policy models. A DataManager and FunctionManager enable defining multi-agent RL GPU-workflows with Python APIs. For more details, please read out white [paper](https://arxiv.org/abs/2108.13976).
#
# The Warpdrive framework comprises several utility functions that help easily implement any (OpenAI-)*gym-style* RL environment, and furthermore, provides quality-of-life tools to train it end-to-end using just a few lines of code. You may familiarize yourself with WarpDrive with the help of these [tutorials](https://github.com/salesforce/warp-drive/tree/master/tutorials).
#
# We invite everyone to **contribute to WarpDrive**, including adding new multi-agent environments, proposing new features and reporting issues on our open source [repository](https://github.com/salesforce/warp-drive).
#
# We have integrated WarpDrive with the [PyTorch Lightning](https://www.lightning.ai/) framework, which greatly reduces the trainer boilerplate code, and improves training modularity and flexibility. It abstracts away most of the engineering pieces of code, so users can focus on research and building models, and iterate on experiments really fast. PyTorch Lightning also provides support for easily running the model on any hardware, performing distributed training, model checkpointing, performance profiling, logging and visualization.
#
# Below, we demonstrate how to use WarpDrive and PyTorch Lightning together to train a game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) where multiple *tagger* agents are trying to run after and tag multiple other *runner* agents. Here's a sample depiction of the game of Tag with $100$ runners and $5$ taggers.
# ![](https://blog.salesforceairesearch.com/content/images/2021/08/same_speed_50fps-1.gif)

# %% [markdown]
# ## Dependencies

# %%
import logging

import torch
from example_envs.tag_continuous.tag_continuous import TagContinuous
from pytorch_lightning import Trainer
from warp_drive.env_wrapper import EnvWrapper
from warp_drive.training.pytorch_lightning import CUDACallback, PerfStatsCallback, WarpDriveModule

# Uncomment below for enabling animation visualizations.
# from example_envs.utils.generate_rollout_animation import generate_tag_env_rollout_animation
# from IPython.display import HTML


# %%
assert torch.cuda.device_count() > 0, "This notebook only runs on a GPU!"

# %%
# Set logger level e.g., DEBUG, INFO, WARNING, ERROR.
logging.getLogger().setLevel(logging.ERROR)

# %% [markdown]
# ## Specify a set of run configurations for your experiments
#
# The run configuration is a dictionary comprising the environment parameters, the trainer and the policy network settings, as well as configurations for saving.
#
# For our experiment, we consider an environment wherein $5$ taggers and $100$ runners play the game of [Tag](https://github.com/salesforce/warp-drive/blob/master/example_envs/tag_continuous/tag_continuous.py) on a $20 \times 20$ plane. The game lasts $200$ timesteps. Each agent chooses it's own acceleration and turn actions at every timestep, and we use mechanics to determine how the agents move over the grid. When a tagger gets close to a runner, the runner is tagged, and is eliminated from the game. For the configuration below, the runners and taggers have the same unit skill levels, or top speeds.
#
# We train the agents using $50$ environments or simulations running in parallel. With WarpDrive, each simulation runs on separate GPU blocks.
#
# There are two separate policy networks used for the tagger and runner agents. Each network is a fully-connected model with two layers each of $256$ dimensions. We use the Advantage Actor Critic (A2C) algorithm for training. WarpDrive also currently provides the option to use the Proximal Policy Optimization (PPO) algorithm instead.

# %%
run_config = dict(
    name="tag_continuous",
    # Environment settings.
    env=dict(
        # number of taggers in the environment
        num_taggers=5,
        # number of runners in the environment
        num_runners=100,
        # length of the (square) grid on which the game is played
        grid_length=20.0,
        # episode length in timesteps
        episode_length=200,
        # maximum acceleration
        max_acceleration=0.1,
        # minimum acceleration
        min_acceleration=-0.1,
        # maximum turn (in radians)
        max_turn=2.35,  # 3pi/4 radians
        # minimum turn (in radians)
        min_turn=-2.35,  # -3pi/4 radians
        # number of discretized accelerate actions
        num_acceleration_levels=10,
        # number of discretized turn actions
        num_turn_levels=10,
        # skill level for the tagger
        skill_level_tagger=1.0,
        # skill level for the runner
        skill_level_runner=1.0,
        # each agent sees the full (or partial) information of the world
        use_full_observation=False,
        # flag to indicate if a runner stays in the game after getting tagged
        runner_exits_game_after_tagged=True,
        # number of other agents each agent can see
        # used in the case use_full_observation is False
        num_other_agents_observed=10,
        # positive reward for a tagger upon tagging a runner
        tag_reward_for_tagger=10.0,
        # negative reward for a runner upon getting tagged
        tag_penalty_for_runner=-10.0,
        # reward at the end of the game for a runner that isn't tagged
        end_of_game_reward_for_runner=1.0,
        # distance margin between a tagger and runner
        # to consider the runner as being 'tagged'
        tagging_distance=0.02,
    ),
    # Trainer settings.
    trainer=dict(
        # number of environment replicas (number of GPU blocks used)
        num_envs=50,
        # total batch size used for training per iteration (across all the environments)
        train_batch_size=10000,
        # total number of episodes to run the training for
        # This can be set arbitrarily high!
        num_episodes=500,
    ),
    # Policy network settings.
    policy=dict(
        runner=dict(
            # flag indicating whether the model needs to be trained
            to_train=True,
            # algorithm used to train the policy
            algorithm="A2C",
            # discount rate
            gamma=0.98,
            # learning rate
            lr=0.005,
            # policy model settings
            model=dict(type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""),
        ),
        tagger=dict(
            to_train=True,
            algorithm="A2C",
            gamma=0.98,
            lr=0.002,
            model=dict(type="fully_connected", fc_dims=[256, 256], model_ckpt_filepath=""),
        ),
    ),
    # Checkpoint saving setting.
    saving=dict(
        # how often (in iterations) to print the metrics
        metrics_log_freq=10,
        # how often (in iterations) to save the model parameters
        model_params_save_freq=5000,
        # base folder used for saving
        basedir="/tmp",
        # experiment name
        name="continuous_tag",
        # experiment tag
        tag="example",
    ),
)

# %% [markdown]
# ## Instantiate the WarpDrive Module
#
# In order to instantiate the WarpDrive module, we first use an environment wrapper to specify that the environment needs to be run on the GPU (via the `use_cuda` flag). Also, agents in the environment can share policy models; so we specify a dictionary to map each policy network model to the list of agent ids using that model.

# %%
# Create a wrapped environment object via the EnvWrapper
# Ensure that env_backend is set to be "pycuda" or "numba"(in order to run on the GPU)
# WarpDrive v2 supports JIT compiled Numba backend now!
env_wrapper = EnvWrapper(
    TagContinuous(**run_config["env"]),
    num_envs=run_config["trainer"]["num_envs"],
    env_backend="pycuda",
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
# ## Visualizing an episode roll-out before training
#
# We have created a helper function (see below) to visualize an episode rollout. Internally, this function uses the WarpDrive module's `fetch_episode_states` API to fetch the data arrays on the GPU for the duration of an entire episode. Specifically, we fetch the state arrays pertaining to agents' x and y locations on the plane and indicators on which agents are still active in the game. Note that this function may be invoked at any time during training, and it will use the state of the policy models at that time to sample actions and generate the visualization.

# %% [markdown]
# The animation below shows a sample realization of the game episode before training, i.e., with randomly chosen agent actions. The $5$ taggers are marked in pink, while the $100$ blue agents are the runners. Both the taggers and runners move around randomly and about half the runners remain at the end of the episode.

# %%
# Uncomment below for enabling animation visualizations.
# anim = generate_tag_env_rollout_animation(wd_module, fps=25)
# HTML(anim.to_html5_video())

# %% [markdown]
# ## Create the Lightning Trainer
#
# Next, we create the trainer for training the WarpDrive model. We add the `performance stats` callbacks to the trainer to view the throughput performance of WarpDrive.

# %%
log_freq = run_config["saving"]["metrics_log_freq"]

# Define callbacks.
cuda_callback = CUDACallback(module=wd_module)
perf_stats_callback = PerfStatsCallback(
    batch_size=wd_module.training_batch_size,
    num_iters=wd_module.num_iters,
    log_freq=log_freq,
)

# Instantiate the PyTorch Lightning trainer with the callbacks.
# Also, set the number of gpus to 1, since this notebook uses just a single GPU.
num_gpus = 1
num_episodes = run_config["trainer"]["num_episodes"]
episode_length = run_config["env"]["episode_length"]
training_batch_size = run_config["trainer"]["train_batch_size"]
num_epochs = int(num_episodes * episode_length / training_batch_size)

trainer = Trainer(
    accelerator="gpu",
    devices=num_gpus,
    callbacks=[cuda_callback, perf_stats_callback],
    max_epochs=num_epochs,
    log_every_n_steps=1,
    reload_dataloaders_every_n_epochs=1,
)

# %%
# Start tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/

# %% [markdown]
# ## Train the WarpDrive Module
#
# Finally, we invoke training.
#
# Note: please scroll up to the tensorboard cell to visualize the curves during training.

# %%
trainer.fit(wd_module)

# %% [markdown]
# ## Visualize an episode-rollout after training

# %%
# Uncomment below for enabling animation visualizations.
# anim = generate_tag_env_rollout_animation(wd_module, fps=25)
# HTML(anim.to_html5_video())

# %% [markdown]
# Note: In the configuration above, we have set the trainer to only train on $500$ rollout episodes, but you can increase the `num_episodes` configuration parameter to train further. As more training happens, the runners learn to escape the taggers, and the taggers learn to chase after the runner. Sometimes, the taggers also collaborate to team-tag runners. A good number of episodes to train on (for the configuration we have used) is $2$M or higher.

# %%
# Finally, close the WarpDrive module to clear up the CUDA memory heap
wd_module.graceful_close()
