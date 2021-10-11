# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: CUDAtorch
#     language: python
#     name: cudatorch
# ---

# %% [markdown]
# # Introduction
#
# In this notebook we will show how to process movielens 1m dataset into a session based recommendation use case and train the Behaviour Sequence Transformer model from [Alibaba](https://arxiv.org/pdf/1905.06874.pdf%C2%A0) with the processed data.
#
# We will use pytorch_lightning and torchmetrics to speed up development.

# %% [markdown]
# # Dependencies

# %%
import math
import os
from urllib.request import urlretrieve
from zipfile import ZipFile

# %%
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn
from torch.utils import data

# %% [markdown]
# ## Settings

# %%
WINDOW_SIZE = 10
DATASET_URL = "http://files.grouplens.org/datasets/movielens/ml-1m.zip"

# %% [markdown]
# # Data

# %% [markdown]
# Lets download the data and read it with pandas

# %%
urlretrieve(DATASET_URL, "movielens.zip")
ZipFile("movielens.zip", "r").extractall()

# %%
users = pd.read_csv(
    "ml-1m/users.dat",
    sep="::",
    names=["user_id", "sex", "age_group", "occupation", "zip_code"],
)

ratings = pd.read_csv(
    "ml-1m/ratings.dat",
    sep="::",
    names=["user_id", "movie_id", "rating", "unix_timestamp"],
)

movies = pd.read_csv("ml-1m/movies.dat", sep="::", names=["movie_id", "title", "genres"])

# %% [markdown]
# ## Preprocessing data

# %% [markdown]
# Making sure all data types are as expected for our preprocessing

# %%
# Movies
movies["year"] = movies["title"].apply(lambda x: x[-5:-1])
movies.year = pd.Categorical(movies.year)
movies["year"] = movies.year.cat.codes

# Users
users.sex = pd.Categorical(users.sex)
users["sex"] = users.sex.cat.codes


users.age_group = pd.Categorical(users.age_group)
users["age_group"] = users.age_group.cat.codes


users.occupation = pd.Categorical(users.occupation)
users["occupation"] = users.occupation.cat.codes


users.zip_code = pd.Categorical(users.zip_code)
users["zip_code"] = users.zip_code.cat.codes

# Ratings
ratings["unix_timestamp"] = pd.to_datetime(ratings["unix_timestamp"], unit="s")


# %% id="6_CC3yYCLxVN"
# Movies
movies["movie_id"] = movies["movie_id"].astype(str)
# Users
users["user_id"] = users["user_id"].astype(str)
# Ratings
ratings["movie_id"] = ratings["movie_id"].astype(str)
ratings["user_id"] = ratings["user_id"].astype(str)

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

for genre in genres:
    movies[genre] = movies["genres"].apply(lambda values: int(genre in values.split("|")))


# %% [markdown] id="0KsqW_4rLxVN"
# ## Transform the movie ratings data into sequences
#
# In order to have a session based recommendation use case we need to create sequences of ratings for each user. The sequence will be ordered based on the time the ratings where created and will provide the model with information on how the user taste evolved over time.
#
# First, let's sort the the ratings data using the `unix_timestamp`, and then group the
# `movie_id` values and the `rating` values by `user_id`.
#
# The output DataFrame will have a record for each `user_id`, with two ordered lists
# (sorted by rating datetime): the movies they have rated, and the ratings of these movies.

# %% id="D5v700zTLxVN"
# Transform all ids and ratings into strings to join them.
# Movies
movies["movie_id"] = movies["movie_id"].astype(str)
# Users
users["user_id"] = users["user_id"].astype(str)
# Ratings
ratings["movie_id"] = ratings["movie_id"].astype(str)
ratings["user_id"] = ratings["user_id"].astype(str)

genres = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

for genre in genres:
    movies[genre] = movies["genres"].apply(lambda values: int(genre in values.split("|")))
ratings_grouped = ratings.sort_values(by=["unix_timestamp"]).groupby("user_id")

ratings_ordered = pd.DataFrame(
    data={
        "user_id": list(ratings_grouped.groups.keys()),
        "movie_ids": list(ratings_grouped.movie_id.apply(list)),
        "ratings": list(ratings_grouped.rating.apply(list)),
    }
)


# %% [markdown] id="USa6rk0eLxVN"
# Now, let's split the `movie_ids` list into a set of sequences of a fixed length.
# We do the same for the `ratings`. Set the `WINDOW_SIZE` variable to change the length
# of the input sequence to the model. You can also change the `step_size` to control the
# number of sequences to generate for each user.
#
# Now that we have the ordered sequence of `movie_id/ratings` per each user we can split them into subsequences of fixed length for our model training.
#
# In each of the subsequences we will use the latest rated movie as target, using previous ratings as input to the model. Here is an image to represent the processing
#
# ![ratings.png](ratings.png)
#
#

# %% id="XdhRJlxULxVN"
# ##########
# Transform a big sequence into fixed length smaller sequences given a window size
# Example:
#     a = [1,2,3,4,5,6,7,8,9,10]
#     create_sequences(a. 5)
#     output:     [[1, 2, 3, 4, 5],
#                  [2, 3, 4, 5, 6],
#                  [3, 4, 5, 6, 7],
#                  [4, 5, 6, 7, 8],
#                  [5, 6, 7, 8, 9],
#                  [6, 7, 8, 9, 10]]
# #########


def create_sequences(values, window_size):
    sequences = []
    start = 0
    pointer = 5
    while pointer <= len(values):
        seq = values[start : start + window_size]
        if len(seq) < window_size:
            seq = values[-window_size:]
            if len(seq) == window_size:
                sequences.append(seq)
            break
        sequences.append(seq)
        start += 1
        pointer += 1
    return sequences


ratings_ordered.movie_ids = ratings_ordered.movie_ids.apply(lambda x: create_sequences(x, WINDOW_SIZE))

ratings_ordered.ratings = ratings_ordered.ratings.apply(lambda x: create_sequences(x, WINDOW_SIZE))

# %% [markdown] id="5dYEduaqLxVN"
# After processing each user history in windows we can explode them to create one row per window.

# %% id="gM5_RBACLxVO"
ratings_data_movies = ratings_ordered[["user_id", "movie_ids"]].explode("movie_ids", ignore_index=True)
ratings_data_rating = ratings_ordered[["ratings"]].explode("ratings", ignore_index=True)
ratings_data_transformed = pd.concat([ratings_data_movies, ratings_data_rating], axis=1)
ratings_data_transformed = ratings_data_transformed.join(users.set_index("user_id"), on="user_id")

ratings_data_transformed.movie_ids = ratings_data_transformed.movie_ids.apply(lambda x: ",".join(x))
ratings_data_transformed.ratings = ratings_data_transformed.ratings.apply(lambda x: ",".join([str(v) for v in x]))

del ratings_data_transformed["zip_code"]

ratings_data_transformed.rename(
    columns={"movie_ids": "sequence_movie_ids", "ratings": "sequence_ratings"},
    inplace=True,
)

# %% [markdown]
# ### Train and test split

# %% [markdown]
# Because we are dealing with user ratings in an ordered manner we will use the latest rating of each user as test set. This will provide us a view on how the model performs on the current time that is being trained on. There is no point evaluating the session based model on random splits as we could suffer from data leakage.

# %% id="0lPMjBoRLxVO"
grouped_ratings = ratings_data_transformed.groupby("user_id")
# Train
train_data = ratings_data_transformed[grouped_ratings.cumcount(ascending=False) > 0]
# Test
test_data = grouped_ratings.tail(1)


# Save primary csv's for later usage
if not os.path.exists("data"):
    os.makedirs("data")

train_data.to_csv("data/train_data.csv", index=False, sep=",")
test_data.to_csv("data/test_data.csv", index=False, sep=",")

# %% [markdown]
# # BST Implementation and training


# %% [markdown]
# We will implement all necessary pytorch code to run our experiment with BST. Lets start with the dataset that will load our sequences

# %% [markdown]
# ## Pytorch dataset

# %%
class MovieDataset(data.Dataset):
    """Movie dataset."""

    def __init__(self, ratings_file, test=False):
        """
        Args:
            csv_file (string): Path to the csv file with user,past,future.
        """
        self.ratings_frame = pd.read_csv(
            ratings_file,
            delimiter=",",
            # iterator=True,
        )
        self.test = test

    def __len__(self):
        return len(self.ratings_frame)

    def __getitem__(self, idx):
        data = self.ratings_frame.iloc[idx]
        user_id = data.user_id

        movie_history = eval(data.sequence_movie_ids)
        movie_history_ratings = eval(data.sequence_ratings)
        target_movie_id = movie_history[-1:][0]
        target_movie_rating = movie_history_ratings[-1:][0]

        movie_history = torch.LongTensor(movie_history[:-1])
        movie_history_ratings = torch.LongTensor(movie_history_ratings[:-1])

        sex = data.sex
        age_group = data.age_group
        occupation = data.occupation

        return (
            user_id,
            movie_history,
            target_movie_id,
            movie_history_ratings,
            target_movie_rating,
            sex,
            age_group,
            occupation,
        )


# %% [markdown]
# # Model

# %% [markdown]
# The model consist on a set of embedding layers for both user and movie ID as well as embedding layers for all their features (genre, sex, occupation etc). It concatenates these embeddings to create vectors for each item of the sequence and pass them throught a Transformer layers that consist only on the encoder block of the transformer. Transformer output is then set to a multilayer feedforward and ends with a single neuron predicting the probability of a user clicking on an item.
#
# We will change the sigmoid for a regular relu in order to predict the rating and do the neccesary changes on embeddings to accomodate movielens dataset.
# This model relies heavily on embeddings for extra feature of the items, in our case this will be for both item data and user data.
#
# ![bst.png](BST.png)

# %%
# Cast to int again to check the maximum vale of each identifier when creating the embedding layers
# Movies
movies["movie_id"] = movies["movie_id"].astype(int)
# Users
users["user_id"] = users["user_id"].astype(int)
# Ratings
ratings["movie_id"] = ratings["movie_id"].astype(int)
ratings["user_id"] = ratings["user_id"].astype(int)


# %%
class BST(pl.LightningModule):
    def __init__(
        self,
        args=None,
    ):
        super().__init__()
        super().__init__()

        self.save_hyperparameters()
        self.args = args
        # -------------------
        # Embedding layers
        # Users
        self.embeddings_user_id = nn.Embedding(int(users.user_id.max()) + 1, int(math.sqrt(users.user_id.max())) + 1)
        # Users features embeddings
        self.embeddings_user_sex = nn.Embedding(len(users.sex.unique()), int(math.sqrt(len(users.sex.unique()))))
        self.embeddings_age_group = nn.Embedding(
            len(users.age_group.unique()), int(math.sqrt(len(users.age_group.unique())))
        )
        self.embeddings_user_occupation = nn.Embedding(
            len(users.occupation.unique()), int(math.sqrt(len(users.occupation.unique())))
        )
        self.embeddings_user_zip_code = nn.Embedding(
            len(users.zip_code.unique()), int(math.sqrt(len(users.sex.unique())))
        )

        user_embedding_size = (
            int(math.sqrt(users.user_id.max()))
            + int(math.sqrt(len(users.sex.unique())))
            + int(math.sqrt(len(users.age_group.unique())))
            + int(math.sqrt(len(users.occupation.unique())))
            + int(math.sqrt(len(users.sex.unique())))
        )
        # Movies
        self.embeddings_movie_id = nn.Embedding(
            int(movies.movie_id.max()) + 1, int(math.sqrt(movies.movie_id.max())) + 1
        )
        self.embeddings_position = nn.Embedding(WINDOW_SIZE, int(math.sqrt(len(movies.movie_id.unique()))) + 1)
        # Movies features embeddings
        genre_vectors = movies[genres].to_numpy()
        self.embeddings_movie_genre = nn.Embedding(genre_vectors.shape[0], genre_vectors.shape[1])

        self.embeddings_movie_genre.weight.requires_grad = False  # Not training genres

        self.embeddings_movie_year = nn.Embedding(len(movies.year.unique()), int(math.sqrt(len(movies.year.unique()))))

        transformer_dim = 63
        self.transfomerlayer = nn.TransformerEncoderLayer(transformer_dim, 3, dropout=0.2)
        transformer_out_dim = WINDOW_SIZE * transformer_dim
        linear_input_dim = transformer_out_dim + user_embedding_size
        self.linear = nn.Sequential(
            nn.Linear(
                linear_input_dim,
                1024,
            ),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
        )
        self.criterion = torch.nn.MSELoss()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def encode_input(self, inputs):
        (
            user_id,
            movie_history,
            target_movie_id,
            movie_history_ratings,
            target_movie_rating,
            sex,
            age_group,
            occupation,
        ) = inputs

        # MOVIES
        movie_history = self.embeddings_movie_id(movie_history)
        target_movie = self.embeddings_movie_id(target_movie_id)

        positions = torch.arange(0, WINDOW_SIZE - 1, 1, dtype=int, device=self.device)
        positions = self.embeddings_position(positions)

        encoded_sequence_movies_with_poistion_and_rating = movie_history + positions  # Yet to multiply by rating

        target_movie = torch.unsqueeze(target_movie, 1)
        transfomer_features = torch.cat((encoded_sequence_movies_with_poistion_and_rating, target_movie), dim=1)

        # USERS
        user_id = self.embeddings_user_id(user_id)

        sex = self.embeddings_user_sex(sex)
        age_group = self.embeddings_age_group(age_group)
        occupation = self.embeddings_user_occupation(occupation)
        user_features = torch.cat((user_id, sex, age_group, occupation), 1)

        return transfomer_features, user_features, target_movie_rating.float()

    def forward(self, batch):
        transfomer_features, user_features, target_movie_rating = self.encode_input(batch)
        transformer_output = self.transfomerlayer(transfomer_features)
        transformer_output = torch.flatten(transformer_output, start_dim=1)

        # Concat with other features
        features = torch.cat((transformer_output, user_features), dim=1)
        output = self.linear(features)
        return output, target_movie_rating

    def training_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        out = out.flatten()
        loss = self.criterion(out, target_movie_rating)

        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse = torch.sqrt(mse)
        self.log("train/mae", mae, on_step=True, on_epoch=False, prog_bar=False)

        self.log("train/rmse", rmse, on_step=True, on_epoch=False, prog_bar=False)

        self.log("train/step_loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        out, target_movie_rating = self(batch)
        out = out.flatten()
        loss = self.criterion(out, target_movie_rating)

        mae = self.mae(out, target_movie_rating)
        mse = self.mse(out, target_movie_rating)
        rmse = torch.sqrt(mse)

        return {"val_loss": loss, "mae": mae.detach(), "rmse": rmse.detach()}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_mae = torch.stack([x["mae"] for x in outputs]).mean()
        avg_rmse = torch.stack([x["rmse"] for x in outputs]).mean()

        self.log("val/loss", avg_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/mae", avg_mae, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/rmse", avg_rmse, on_step=False, on_epoch=True, prog_bar=False)

    def test_epoch_end(self, outputs):
        users = torch.cat([x["users"] for x in outputs])
        y_hat = torch.cat([x["top14"] for x in outputs])
        users = users.tolist()
        y_hat = y_hat.tolist()

        data = {"users": users, "top14": y_hat}
        df = pd.DataFrame.from_dict(data)
        df.to_csv("lightning_logs/predict.csv", index=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=0.0005)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", type=float, default=0.01)
        return parser

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        print("Loading datasets")
        self.train_dataset = MovieDataset("data/train_data.csv")
        self.val_dataset = MovieDataset("data/test_data.csv")
        self.test_dataset = MovieDataset("data/test_data.csv")
        print("Done")

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=os.cpu_count(),
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=os.cpu_count(),
        )


# %% [markdown]
# # Training

# %% [markdown]
# With pytorch lightning is easy to start training. We will start tensorboard to monitor how training goes and call the `fit` function.

# %%
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs

# %%
model = BST()
trainer = pl.Trainer(gpus=1, max_epochs=5)
trainer.fit(model)

# %%
