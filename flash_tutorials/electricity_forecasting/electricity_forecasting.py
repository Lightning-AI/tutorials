# %% [markdown]
# In this tutorial we'll look at using [Lightning Flash](https://github.com/Lightning-AI/lightning-flash) and it's
# integration with [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) for autoregressive modelling of
# electricity prices using [the N-BEATS model](https://arxiv.org/abs/1905.10437).
# We'll start by using N-BEATS to uncover daily patterns (seasonality) from hourly observations and then show how we can
# resample daily averages to uncover weekly patterns too.
#
# Along the way, we'll see how the built-in tools from PyTorch Lightning, like the learning rate finder, can be used
# seamlessly with Flash to help make the process of putting a model together as smooth as possible.

# %%

import os
from typing import Any, Dict

import flash
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flash.core.data.utils import download_data
from flash.core.integrations.pytorch_forecasting import convert_predictions
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData

DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")

# %% [markdown]
# ## Loading the data
#
# We'll use the Spanish hourly energy demand generation and weather data set from Kaggle:
# https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
#
# First, download the data:

# %%
download_data("https://pl-flash-data.s3.amazonaws.com/kaggle_electricity.zip", DATASET_PATH)

# %% [markdown]
# ## Data loading
#
# To load the data, we start by loading the CSV file into a pandas DataFrame:

# %%
df_energy_hourly = pd.read_csv(f"{DATASET_PATH}/energy_dataset.csv", parse_dates=["time"])

# %% [markdown]
# Before we can load the data into Flash, there are a few preprocessing steps we need to take.
# The first preprocessing step is to set the `time` field as the index (formatted as a datetime).
# The second step is to resample the data to the desired frequency in case it is different from the desired observation
# frequency.
# Since we are performing autoregressive modelling, we can remove all columns except for `"price actual"`.
#
# For the third preprocessing step, we need to create a "time_idx" column.
# The "time_idx" column should contain integers corresponding to the observation index (e.g. in our case the difference
# between two "time_idx" values is the number of hours between the observations).
# To do this we convert the datetime to an index by taking the nanoseconds value and dividing by the number of
# nanoseconds in a single unit of our chosen frequency.
# We then subtract the minimum value so it starts at zero (although it would still work without this step).
#
# The Flash `TabularForecastingData` (which uses the `TimeSeriesDataSet` from PyTorch Forecasting internally) also
# supports loading data from multiple time series (e.g. you may have electricity data from multiple countries).
# To indicate that our data is all from the same series, we add a `constant` column with a constant value of zero.
#
# Here's the full preprocessing function:

# %%


def preprocess(df: pd.DataFrame, frequency: str = "1H") -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"], utc=True, infer_datetime_format=True)
    df.set_index("time", inplace=True)

    df = df.resample(frequency).mean()

    df = df.filter(["price actual"])

    df["time_idx"] = (df.index.view(int) / pd.Timedelta(frequency).value).astype(int)
    df["time_idx"] -= df["time_idx"].min()

    df["constant"] = 0

    return df


df_energy_hourly = preprocess(df_energy_hourly)

# %% [markdown]
# ## Creating the Flash DataModule
#
# Now, we can create a `TabularForecastingData`.
# The role of the `TabularForecastingData` is to split up our time series into windows which include a region to encode
# (of size `max_encoder_length`) and a region to predict (of size `max_prediction_length`) which will be used to compute
# the loss.
# The size of the prediction window should be chosen depending on the kinds of trends we would like our model to
# uncover.
# In our case, we are interested in how electricity prices change throughout the day, so a one day prediction window
# (`max_prediction_length = 24`) makes sense here.
# The size of the encoding window can vary, however, in the [N-BEATS paper](https://arxiv.org/abs/1905.10437) the
# authors suggest using an encoder length of between two and ten times the prediction length.
# We therefore choose two days (`max_encoder_length = 48`) as the encoder length.

# %%
max_prediction_length = 24
max_encoder_length = 24 * 2

training_cutoff = df_energy_hourly["time_idx"].max() - max_prediction_length

datamodule = TabularForecastingData.from_data_frame(
    time_idx="time_idx",
    target="price actual",
    group_ids=["constant"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["price actual"],
    train_data_frame=df_energy_hourly[df_energy_hourly["time_idx"] <= training_cutoff],
    val_data_frame=df_energy_hourly,
    batch_size=256,
)

# %% [markdown]
# ## Creating the Flash Task
#
# Now, we're ready to create a `TabularForecaster`.
# The N-BEATS model has two primary hyper-parameters:`"widths"`, and `"backcast_loss_ratio"`.
# In the [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.nbeats.NBeats.html),
# the authors recommend using `"widths"` of `[32, 512]`.
# In order to prevent overfitting with smaller datasets, a good rule of thumb is to limit the number of parameters of
# your model.
# For this reason, we use `"widths"` of `[16, 256]`.
#
# To understand the `"backcast_loss_ratio"`, let's take a look at this diagram of the model taken from
# [the arXiv paper](https://arxiv.org/abs/1905.10437):
#
# ![N-BEATS diagram](diagram.png)
#
# Each 'block' within the N-BEATS architecture includes a forecast output and a backcast which can each yield their own
# loss.
# The `"backcast_loss_ratio"` is the ratio of the backcast loss to the forecast loss.
# A value of `1.0` means that the loss function is simply the sum of the forecast and backcast losses.

# %%
model = TabularForecaster(
    datamodule.parameters, backbone="n_beats", backbone_kwargs={"widths": [16, 256], "backcast_loss_ratio": 1.0}
)

# %% [markdown]
# ## Finding the learning rate
#
# Tabular models can be particularly sensitive to the choice of learning rate.
# Helpfully, PyTorch Lightning provides a built-in learning rate finder that suggests a suitable learning rate
# automatically.
# To use it, we first create our Trainer.
# We apply gradient clipping (a common technique for tabular tasks) with ``gradient_clip_val=0.01`` in order to help
# prevent our model from over-fitting.
# Here's how to find the learning rate:

# %%
trainer = flash.Trainer(
    max_epochs=3,
    gpus=int(torch.cuda.is_available()),
    gradient_clip_val=0.01,
)

res = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-5)
print(f"Suggested learning rate: {res.suggestion()}")
res.plot(show=True, suggest=True).show()

# %% [markdown]
# Once the suggest learning rate has been found, we can update our model with it:

# %%
model.learning_rate = res.suggestion()

# %% [markdown]
# ## Training the model
# Now all we have to do is train the model!

# %%
trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# ## Plot the interpretation
#
# An important feature of the N-BEATS model is that it can be configured to produce an interpretable prediction that is
# split into both a low frequency (trend) component and a high frequency (seasonality) component.
# For hourly observations, we might expect the trend component to show us how electricity prices are changing from one
# day to the next (for example, whether prices were generally higher or lower than yesterday).
# In contrast, the seasonality component would be expected to show us the general pattern in prices through the day
# (for example, if there is typically a peak in price around lunch time or a drop at night).
#
# It is often useful to visualize this decomposition and the `TabularForecaster` makes it simple.
# First, we load the best model from our training run and generate some predictions.
# Next, we convert the predictions to the format expected by PyTorch Forecasting using the `convert_predictions` utility
# function.
# Finally, we plot the interpretation using the `pytorch_forecasting_model` attribute.
# Here's the full function:

# %%


def plot_interpretation(model_path: str, predict_df: pd.DataFrame, parameters: Dict[str, Any]):
    model = TabularForecaster.load_from_checkpoint(model_path)
    datamodule = TabularForecastingData.from_data_frame(
        parameters=parameters,
        predict_data_frame=predict_df,
        batch_size=256,
    )
    trainer = flash.Trainer(gpus=int(torch.cuda.is_available()))
    predictions = trainer.predict(model, datamodule=datamodule)
    predictions, inputs = convert_predictions(predictions)
    model.pytorch_forecasting_model.plot_interpretation(inputs, predictions, idx=0)
    plt.show()


# %% [markdown]
# And now we run the function to plot the trend and seasonality curves:

# %%
# Todo: Make sure to uncomment the line below if you want to run predictions and visualize the graph
# plot_interpretation(trainer.checkpoint_callback.best_model_path, df_energy_hourly, datamodule.parameters)

# %% [markdown]
# It worked! The plot shows that the `TabularForecaster` does a reasonable job of modelling the time series and also
# breaks it down into a trend component and a seasonality component (in this case showing daily fluctuations in
# electricity prices).
#
# ## Bonus: Weekly trends
#
# The type of seasonality that the model learns to detect is dictated by the frequency of observations and the length of
# the encoding / prediction window.
# We might imagine that our pipeline could be changed to instead uncover weekly trends if we resample daily
# observations from our data instead of hourly.
#
# We can use our preprocessing function to do this.
# First, we load the data as before then preprocess it (this time setting `frequency = "1D"`).

# %%
df_energy_daily = pd.read_csv(f"{DATASET_PATH}/energy_dataset.csv", parse_dates=["time"])
df_energy_daily = preprocess(df_energy_daily, frequency="1D")

# %% [markdown]
# Now let's create our `TabularForecastingData` as before, this time with a four week encoding window and a one week
# prediction window.

# %%
max_prediction_length = 1 * 7
max_encoder_length = 4 * 7

training_cutoff = df_energy_daily["time_idx"].max() - max_prediction_length

datamodule = TabularForecastingData.from_data_frame(
    time_idx="time_idx",
    target="price actual",
    group_ids=["constant"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["price actual"],
    train_data_frame=df_energy_daily[df_energy_daily["time_idx"] <= training_cutoff],
    val_data_frame=df_energy_daily,
    batch_size=256,
)

# %% [markdown]
# Now it's time to create a new model and trainer.
# We run for 24 times the number of epochs this time as we now have around 1/24th of the number of observations.
# This time, instead of using the learning rate finder we just set the learning rate manually:

# %%
model = TabularForecaster(
    datamodule.parameters,
    backbone="n_beats",
    backbone_kwargs={"widths": [16, 256], "backcast_loss_ratio": 1.0},
    learning_rate=5e-4,
)

trainer = flash.Trainer(
    max_epochs=3 * 24,
    check_val_every_n_epoch=24,
    gpus=int(torch.cuda.is_available()),
    gradient_clip_val=0.01,
)

# %% [markdown]
# Finally, we train the new model:

# %%
trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# Now let's look at what it learned:

# %%
# Todo: Make sure to uncomment the line below if you want to run predictions and visualize the graph
# plot_interpretation(trainer.checkpoint_callback.best_model_path, df_energy_daily, datamodule.parameters)

# %% [markdown]
# Success! We can now also see weekly trends / seasonality uncovered by our new model.
#
# ## Closing thoughts and next steps!
#
# This tutorial has shown how Flash and PyTorch Forecasting can be used to train state-of-the-art auto-regressive
# forecasting models (such as N-BEATS).
# We've seen how we can influence the kinds of trends and patterns uncovered by the model by resampling the data and
# changing the hyper-parameters.
#
# There are plenty of ways you could take this tutorial further.
# For example, you could try a more complex model, such as the
# [temporal fusion transformer](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.temporal_fusion_transformer.TemporalFusionTransformer.html),
# which can handle additional inputs (the kaggle data set we used also includes weather data).
#
# Alternatively, if you want to be a bit more adventurous, you could look at
# [some of the other problems that can solved with Lightning Flash](https://lightning-flash.readthedocs.io/en/stable/?badge=stable).
