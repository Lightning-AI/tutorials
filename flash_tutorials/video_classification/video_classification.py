# %% [markdown]
# ## Video Classification
#
#

# %% [markdown]
# ## Downloading the data
#
# Run this command to download the the sample dataset
# You'll need to download the kaggle.json file after generating the API token.
#
# kaggle competitions download -c deepfake-detection-challenge
# unzip deepfake-detection-challenge.zip

# %% [markdown]
# ## Preprocessing the data for Lightning Flash
#
# The full dataset for the challenge in ~500 GB in size, so we'll use the provided ``train_sample_videos`` to demo the pre-processing script.

# %%
# import os
# import json
# import shutil


# base_folder = 'train_sample_videos'
# label_file = 'metadata.json'

# fake_output_dir = 'train_sample_videos/fake'
# real_output_dir = 'train_sample_videos/real'

# f = open(os.path.join(base_folder, label_file))
# data = json.load(f)

# for video in data.keys():
#     source_video = os.path.join(base_folder, video)

#     if data[video]['label'] == 'FAKE':
#         dest_video = os.path.join(fake_output_dir, video)
#     elif data[video]['label'] == 'REAL':
#         dest_video = os.path.join(real_output_dir, video)

#     try:
#         shutil.move(source_video, dest_video)
#         print(source_video + ' -> ' + dest_video)
#     except FileNotFoundError:
#         print('Skipping ' + source_video)

# f.close()


# %% [markdown]
# ### Dataset statistics and split
#
# The ``train_sample_videos`` contains 323 fake videos and 77 real videos. The actual training set contains 99992 fake and 19154 real videos.
# We will split the dataset into dev, val and test with a 80, 10 and 10 split. Since the dataset is unbalanced between the classes, we'll split the 2 classes separately in this ratio.


# %%
# import random


# random.seed(1234)

# fake_video_files = [
#     f for f in os.listdir(fake_output_dir)
#     if os.path.isfile(os.path.join(fake_output_dir, f)) and '.mp4' in f
# ]
# real_video_files = [
#     f for f in os.listdir(real_output_dir)
#     if os.path.isfile(os.path.join(real_output_dir, f)) and '.mp4' in f
# ]

# random.shuffle(fake_video_files)
# random.shuffle(real_video_files)

# fake_val_test_len = len(fake_video_files) // 10
# real_val_test_len = len(real_video_files) // 10

# fake_val_videos = fake_video_files[:len(fake_video_files) // 10]
# fake_test_videos = fake_video_files[len(fake_video_files) // 10 : 2 * len(fake_video_files) // 10]
# fake_train_videos = fake_video_files[2 * len(fake_video_files) // 10:]

# real_val_videos = real_video_files[:len(real_video_files) // 10]
# real_test_videos = real_video_files[len(real_video_files) // 10 : 2 * len(real_video_files) // 10]
# real_train_videos = real_video_files[2 * len(real_video_files) // 10:]

# def transfer_files(video_list, source_folder, dest_folder):
#     for video in video_list:
#         source_video = os.path.join(source_folder, video)
#         dest_video = os.path.join(dest_folder, video)

#         try:
#             shutil.move(source_video, dest_video)
#             print(source_video + ' -> ' + dest_video)
#         except FileNotFoundError:
#             print('Skipping ' + source_video)


# transfer_files(fake_val_videos, fake_output_dir, os.path.join(base_folder, 'val/fake'))
# transfer_files(fake_test_videos, fake_output_dir, os.path.join(base_folder, 'test/fake'))
# transfer_files(fake_train_videos, fake_output_dir, os.path.join(base_folder, 'train/fake'))

# transfer_files(real_val_videos, real_output_dir, os.path.join(base_folder, 'val/real'))
# transfer_files(real_test_videos, real_output_dir, os.path.join(base_folder, 'test/real'))
# transfer_files(real_train_videos, real_output_dir, os.path.join(base_folder, 'train/real'))


# %% [markdown]
# ### Training a video classifier
#
# Now we use the example code from Flash to start training our video classifier.

import flash

# %%
import torch
from flash.video import VideoClassificationData, VideoClassifier

datamodule = VideoClassificationData.from_folders(
    train_folder="train_sample_videos/train",
    val_folder="train_sample_videos/val",
    clip_sampler="uniform",
    clip_duration=1,
    decode_audio=False,
    batch_size=1,
)

model = VideoClassifier(backbone="x3d_xs", num_classes=datamodule.num_classes, pretrained=True)

trainer = flash.Trainer(max_epochs=3, gpus=torch.cuda.device_count())
trainer.finetune(model, datamodule=datamodule, strategy="freeze")

datamodule = VideoClassificationData.from_folders(predict_folder="train_sample_videos/test", batch_size=1)
predictions = trainer.predict(model, datamodule=datamodule)
print(predictions)
