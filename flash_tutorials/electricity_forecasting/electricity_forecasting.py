# %% [markdown]
# In this tutorial we'll look at using [Lightning Flash](https://github.com/PyTorchLightning/lightning-flash) and it's
# integration with [PyTorch Forecasting](https://github.com/jdb78/pytorch-forecasting) for autoregressive modelling of
# electricity prices using the NBeats model.
# We'll start by using NBeats to uncover daily patterns (seasonality) from hourly observations and then show how we can
# resample daily averages to uncover weekly patterns too.
#
# Along the way, we'll see how the built-in tools from PyTorch Lightning, like the learning rate finder, can be used
# seamlessly with Flash to help make the process of putting a model together as smooth as possible.

# %%

import flash
import matplotlib.pyplot as plt
import pandas as pd
import torch
from flash.core.data.utils import download_data
from flash.core.integrations.pytorch_forecasting import convert_predictions
from flash.tabular.forecasting import TabularForecaster, TabularForecastingData

# %% [markdown]
# ## Loading the data
#
# We'll use the Spanish electricity pricing data from this Kaggle data set:
# https://www.kaggle.com/nicholasjhana/energy-consumption-generation-prices-and-weather
#
# First, download the data:

# %%
download_data("https://pl-flash-data.s3.amazonaws.com/kaggle_electricity.zip", "./data")

# %% [markdown]
# ## Data loading
#
# To load the data, we start by loading the CSV file into a pandas DataFrame:

# %%
df_energy_hourly = pd.read_csv("./data/energy_dataset.csv", parse_dates=["time"])

# %% [markdown]
# In order to load the data into Flash, there are a few preprocessing steps we need to take.
# Next, we set the `time` field as the index (formatted as a datetime):

# %%
df_energy_hourly["time"] = pd.to_datetime(df_energy_hourly["time"], utc=True, infer_datetime_format=True)
df_energy_hourly = df_energy_hourly.set_index("time")

# %% [markdown]
# Since we are performing autoregressive modelling, we remove all columns except for `"price actual"`:

# %%
df_energy_hourly = df_energy_hourly.filter(["price actual"])

# %% [markdown]
# ## Creating the Flash DataModule
#
# Before we can load our data with Flash, we first need to create "time_idx" column.
# The "time_idx" column should contain integers corresponding to the observation index (e.g. in our case the difference
# between two "time_idx" values is the number of hours between the observations).
# To do this we convert the datetime to an index (by taking the nanoseconds value and dividing by the number of
# nanoseconds in an hour) then subtract the minimum value so it starts at zero:

# %%
df_energy_hourly["time_idx"] = (df_energy_hourly.index.view(int) / (1e9 * 60 * 60)).astype(int)
df_energy_hourly["time_idx"] -= df_energy_hourly["time_idx"].min()

# %% [markdown]
# The Flash `TabularForecastingData` (which uses the `TimeSeriesDataSet` from PyTorch Forecasting internally) also
# supports loading data from multiple time series (e.g. you may have electricity data from multiple countries).
# To indicate that our data is all from the same series, we add a `constant` column with a constant value of zero:

# %%
df_energy_hourly["constant"] = 0

# %% [markdown]
# Now we can create our `TabularForecastingData`.
# We choose two days (48 hours) as the encoder length (`max_encoder_length`) and one day (24 hours) as the prediction
# length (`max_prediction_length`).

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
    train_data_frame=df_energy_hourly[lambda x: x.time_idx <= training_cutoff],
    val_data_frame=df_energy_hourly,
    batch_size=512,
    num_workers=2,
)

# %% [markdown]
# ## Creating the Flash Task
#
# No we're ready to create our `TabularForecaster`.
# `"widths"` and `"backcast_loss_ratio"` are hyper-parameters of the NBeats model that we are using.
# The [PyTorch Forecasting Documentation](https://pytorch-forecasting.readthedocs.io/en/latest/api/pytorch_forecasting.models.nbeats.NBeats.html)
# recommends `"widths"` of `[32, 512]` but since our data here is quite simple, we halve it to try to prevent
# overfitting.
#
# To understand the `"backcast_loss_ratio"`, take a look at this diagram of the model taken from
# [the arXiv paper](https://arxiv.org/abs/1905.10437):
#
# <center width="100%" style="padding:10px"><img src="diagram.png" width="250px"></center>
#
# Each 'block' within the NBeats architecture includes a forecast output and a backcast which can each yield their own
# loss.
# The `"backcast_loss_ratio"` is the ratio of the backcast loss to the forecast loss, a value of `0.1` here meaning that
# the forecast loss will be weighted ten times the backcast loss.

# %%
model = TabularForecaster(
    datamodule.parameters, backbone="n_beats", backbone_kwargs={"widths": [16, 256], "backcast_loss_ratio": 0.1}
)

# %% [markdown]
# ## Finding the learning rate
#
# Tabular models can be particularly sensitive to the choice of learning rate.
# We can employ the learning rate finder from PyTorch Lightning to suggest a suitable learning rate automatically like
# this:

# %%
trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count(), gradient_clip_val=0.01)

res = trainer.tuner.lr_find(model, datamodule=datamodule, min_lr=1e-5)
print(f"Suggested learning rate: {res.suggestion()}")
res.plot(show=True, suggest=True).show()

# %% [markdown]
# And update our model with the suggested learning rate:

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
# An important feature of the NBeats model is that it can be configured to detect both low frequency 'trends' and high
# frequency 'seasonality' in a single model.
# PyTorch Forecasting includes the ability to plot this decomposition, a feature we can also use with our Flash
# `TabularForecaster`.
#
# First, we load the best model from our training run and generate some predictions:

# %%
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = TabularForecaster.load_from_checkpoint(best_model_path)

predictions = best_model.predict(df_energy_hourly)

# %% [markdown]
# Next, we convert the predictions to the format expected by PyTorch Forecasting using the `convert_predictions` utility
# function.

# %%
predictions, inputs = convert_predictions(predictions)

# %% [markdown]
# Finally, we plot the interpretation using the `pytorch_forecasting_model` attribute:

# %%
best_model.pytorch_forecasting_model.plot_interpretation(inputs, predictions, idx=0)
plt.show()

# %% [markdown]
# It worked! The plot shows that the `TabularForecaster` does a reasonable job of modelling the time series and also
# breaks it down into a trend component and a seasonality component (in this case showing daily fluctuations in
# electricity prices).
#
# ## Bonus: weekly trends
#
# The type of seasonality that the model learns to detect is dictated by the frequency of observations and the length of
# the encoding / prediction window.
# We might imagine that our pipeline could be changed to instead uncover weekly trends if we resample daily
# observations from our data.
#
# We can use pandas to do this.
# First, we load the data as before and create the index:

# %%
df_energy_daily = pd.read_csv("./data/energy_dataset.csv", parse_dates=["time"])

df_energy_daily["time"] = pd.to_datetime(df_energy_daily["time"], utc=True, infer_datetime_format=True)
df_energy_daily = df_energy_daily.set_index("time")

# %% [markdown]
# Next, we use the `resample` method to give us the mean value for each day (you could experiment with taking a max
# here):

# %%
df_energy_daily = df_energy_daily.resample("D").mean()

# %% [markdown]
# Now let's create our `TabularForecastingData` as before, this time with an four week encoding window and a two week
# prediction window.

# %%
df_energy_daily = df_energy_daily.filter(["price actual"])

df_energy_daily["time_idx"] = (df_energy_daily.index.view(int) / (1e9 * 60 * 60 * 24)).astype(int)
df_energy_daily["time_idx"] -= df_energy_daily["time_idx"].min()

df_energy_daily["constant"] = 0

max_prediction_length = 2 * 7
max_encoder_length = 4 * 7

training_cutoff = df_energy_daily["time_idx"].max() - max_prediction_length

datamodule = TabularForecastingData.from_data_frame(
    time_idx="time_idx",
    target="price actual",
    group_ids=["constant"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    time_varying_unknown_reals=["price actual"],
    train_data_frame=df_energy_daily[lambda x: x.time_idx <= training_cutoff],
    val_data_frame=df_energy_daily,
    batch_size=512,
    num_workers=2,
)

# %% [markdown]
# Now it's time to create a new model and trainer.
# We reduce the widths even further and run for many more epochs this time as we now have even fewer observations.
# This time, instead of using the learning rate finder we just set the learning rate manually:

# %%
model = TabularForecaster(
    datamodule.parameters,
    backbone="n_beats",
    backbone_kwargs={"widths": [8, 128], "backcast_loss_ratio": 0.1},
    learning_rate=5e-4,
)

trainer = flash.Trainer(
    max_epochs=3 * 24,
    check_val_every_n_epoch=24,
    gpus=torch.cuda.device_count(),
    gradient_clip_val=0.01,
)

# %% [markdown]
# Finally, we train the new model:

# %%
trainer.fit(model, datamodule=datamodule)

# %% [markdown]
# Now let's look at what it learned:

# %%
best_model_path = trainer.checkpoint_callback.best_model_path
best_model = TabularForecaster.load_from_checkpoint(best_model_path)

predictions, inputs = convert_predictions(best_model.predict(df_energy_daily))

best_model.pytorch_forecasting_model.plot_interpretation(inputs, predictions, idx=0)
plt.show()

# %% [markdown]
# Success! We can now also see weekly trends / seasonality uncovered by our new model.
#
# ## Closing thoughts and next steps!
#
# This tutorial has shown how Flash and PyTorch Forecasting can be used to train state-of-the-art auto-regressive
# forecasting models (such as NBeats).
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
