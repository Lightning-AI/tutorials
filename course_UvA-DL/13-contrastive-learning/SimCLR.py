# %% [markdown]
# <div class="center-wrapper"><div class="video-wrapper"><iframe src="https://www.youtube.com/embed/waVZDFR-06U" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div></div>
# Methods for self-supervised learning try to learn as much as possible from the data alone, so it can quickly be finetuned for a specific classification task.
# The benefit of self-supervised learning is that a large dataset can often easily be obtained.
# For instance, if we want to train a vision model on semantic segmentation for autonomous driving, we can collect large amounts of data by simply installing a camera in a car, and driving through a city for an hour.
# In contrast, if we would want to do supervised learning, we would have to manually label all those images before training a model.
# This is extremely expensive, and would likely take a couple of months to manually label the same amount of data.
# Further, self-supervised learning can provide an alternative to transfer learning from models pretrained on ImageNet since we could pretrain a model on a specific dataset/situation, e.g. traffic scenarios for autonomous driving.
#
# Within the last two years, a lot of new approaches have been proposed for self-supervised learning, in particular for images, that have resulted in great improvements over supervised models when few labels are available.
# The subfield that we will focus on in this tutorial is contrastive learning.
# Contrastive learning is motivated by the question mentioned above: how are images different from each other?
# Specifically, contrastive learning methods train a model to cluster an image and its slightly augmented version in latent space, while the distance to other images should be maximized.
# A very recent and simple method for this is [SimCLR](https://arxiv.org/abs/2006.10029), which is visualized below (figure credit - [Ting Chen et al. ](https://simclr.github.io/)).
#
# <center width="100%"> ![simclr contrastive learning](simclr_contrastive_learning.png){width="500px"} </center>
#
# The general setup is that we are given a dataset of images without any labels, and want to train a model on this data such that it can quickly adapt to any image recognition task afterward.
# During each training iteration, we sample a batch of images as usual.
# For each image, we create two versions by applying data augmentation techniques like cropping, Gaussian noise, blurring, etc.
# An example of such is shown on the left with the image of the dog.
# We will go into the details and effects of the chosen augmentation techniques later.
# On those images, we apply a CNN like ResNet and obtain as output a 1D feature vector on which we apply a small MLP.
# The output features of the two augmented images are then trained to be close to each other, while all other images in that batch should be as different as possible.
# This way, the model has to learn to recognize the content of the image that remains unchanged under the data augmentations, such as objects which we usually care about in supervised tasks.
#
# We will now implement this framework ourselves and discuss further details along the way.
# Let's first start with importing our standard libraries below:

# %%
import os
import urllib.request
from copy import deepcopy
from urllib.error import HTTPError

import lightning as L
import matplotlib
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from torchvision import transforms
from torchvision.datasets import STL10
from tqdm.notebook import tqdm

plt.set_cmap("cividis")
# %matplotlib inline
matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.set()

# Import tensorboard
# %load_ext tensorboard

# Path to the folder where the datasets are/should be downloaded (e.g. CIFAR10)
DATASET_PATH = os.environ.get("PATH_DATASETS", "data/")
# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models/ContrastiveLearning/")
# In this notebook, we use data loaders with heavier computational processing. It is recommended to use as many
# workers as possible in a data loader, which corresponds to the number of CPU cores
NUM_WORKERS = os.cpu_count()

# Setting the seed
L.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

# %% [markdown]
# As in many tutorials before, we provide pre-trained models.
# Note that those models are slightly larger as normal (~100MB overall) since we use the default ResNet-18 architecture.
# If you are running this notebook locally, make sure to have sufficient disk space available.

# %%
# Github URL where saved models are stored for this tutorial
base_url = "https://raw.githubusercontent.com/phlippe/saved_models/main/tutorial17/"
# Files to download
pretrained_files = [
    "SimCLR.ckpt",
    "ResNet.ckpt",
    "tensorboards/SimCLR/events.out.tfevents.SimCLR",
    "tensorboards/classification/ResNet/events.out.tfevents.ResNet",
]
pretrained_files += [f"LogisticRegression_{size}.ckpt" for size in [10, 20, 50, 100, 200, 500]]
# Create checkpoint path if it doesn't exist yet
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# For each file, check whether it already exists. If not, try downloading it.
for file_name in pretrained_files:
    file_path = os.path.join(CHECKPOINT_PATH, file_name)
    if "/" in file_name:
        os.makedirs(file_path.rsplit("/", 1)[0], exist_ok=True)
    if not os.path.isfile(file_path):
        file_url = base_url + file_name
        print(f"Downloading {file_url}...")
        try:
            urllib.request.urlretrieve(file_url, file_path)
        except HTTPError as e:
            print(
                "Something went wrong. Please try to download the file from the GDrive folder, or contact the author with the full output including the following error:\n",
                e,
            )

# %% [markdown]
# ## SimCLR
#
# We will start our exploration of contrastive learning by discussing the effect of different data augmentation techniques, and how we can implement an efficient data loader for such.
# Next, we implement SimCLR with PyTorch Lightning, and finally train it on a large, unlabeled dataset.

# %% [markdown]
# ### Data Augmentation for Contrastive Learning
#
# To allow efficient training, we need to prepare the data loading such that we sample two different, random augmentations for each image in the batch.
# The easiest way to do this is by creating a transformation that, when being called, applies a set of data augmentations to an image twice.
# This is implemented in the class `ContrastiveTransformations` below:


# %%
class ContrastiveTransformations:
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


# %% [markdown]
# The contrastive learning framework can easily be extended to have more _positive_ examples by sampling more than two augmentations of the same image.
# However, the most efficient training is usually obtained by using only two.
#
# Next, we can look at the specific augmentations we want to apply.
# The choice of the data augmentation to use is the most crucial hyperparameter in SimCLR since it directly affects how the latent space is structured, and what patterns might be learned from the data.
# Let's first take a look at some of the most popular data augmentations (figure credit - [Ting Chen and Geoffrey Hinton](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)):
#
# <center width="100%"><img src="simclr_data_augmentations.jpg" width="800px" style="padding-top: 10px; padding-bottom: 10px"></center>
#
# All of them can be used, but it turns out that two augmentations stand out in their importance: crop-and-resize, and color distortion.
# Interestingly, however, they only lead to strong performance if they have been used together as discussed by [Ting Chen et al. ](https://arxiv.org/abs/2006.10029) in their SimCLR paper.
# When performing randomly cropping and resizing, we can distinguish between two situations: (a) cropped image A provides a local view of cropped image B, or (b) cropped images C and D show neighboring views of the same image (figure credit - [Ting Chen and Geoffrey Hinton](https://ai.googleblog.com/2020/04/advancing-self-supervised-and-semi.html)).
#
# <center width="100%"><img src="crop_views.svg" width="400px" style="padding-top: 20px; padding-bottom: 0px"></center>
#
# While situation (a) requires the model to learn some sort of scale invariance to make crops A and B similar in latent space, situation (b) is more challenging since the model needs to recognize an object beyond its limited view.
# However, without color distortion, there is a loophole that the model can exploit, namely that different crops of the same image usually look very similar in color space.
# Consider the picture of the dog above.
# Simply from the color of the fur and the green color tone of the background, you can reason that two patches belong to the same image without actually recognizing the dog in the picture.
# In this case, the model might end up focusing only on the color histograms of the images, and ignore other more generalizable features.
# If, however, we distort the colors in the two patches randomly and independently of each other, the model cannot rely on this simple feature anymore.
# Hence, by combining random cropping and color distortions, the model can only match two patches by learning generalizable representations.
#
# Overall, for our experiments, we apply a set of 5 transformations following the original SimCLR setup: random horizontal flip, crop-and-resize, color distortion, random grayscale, and gaussian blur.
# In comparison to the [original implementation](https://github.com/google-research/simclr), we reduce the effect of the color jitter slightly (0.5 instead of 0.8 for brightness, contrast, and saturation, and 0.1 instead of 0.2 for hue).
# In our experiments, this setting obtained better performance and was faster and more stable to train.
# If, for instance, the brightness scale highly varies in a dataset, the
# original settings can be more beneficial since the model can't rely on
# this information anymore to distinguish between images.

# %%
contrast_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

# %% [markdown]
# After discussing the data augmentation techniques, we can now focus on the dataset.
# In this tutorial, we will use the [STL10 dataset](https://cs.stanford.edu/~acoates/stl10/), which, similarly to CIFAR10, contains images of 10 classes: airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck.
# However, the images have a higher resolution, namely $96\times 96$ pixels, and we are only provided with 500 labeled images per class.
# Additionally, we have a much larger set of $100,000$ unlabeled images which are similar to the training images but are sampled from a wider range of animals and vehicles.
# This makes the dataset ideal to showcase the benefits that self-supervised learning offers.
#
# Luckily, the STL10 dataset is provided through torchvision.
# Keep in mind, however, that since this dataset is relatively large and has a considerably higher resolution than CIFAR10, it requires more disk space (~3GB) and takes a bit of time to download.
# For our initial discussion of self-supervised learning and SimCLR, we
# will create two data loaders with our contrastive transformations above:
# the `unlabeled_data` will be used to train our model via contrastive
# learning, and `train_data_contrast` will be used as a validation set in
# contrastive learning.

# %%
unlabeled_data = STL10(
    root=DATASET_PATH,
    split="unlabeled",
    download=True,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)
train_data_contrast = STL10(
    root=DATASET_PATH,
    split="train",
    download=True,
    transform=ContrastiveTransformations(contrast_transforms, n_views=2),
)

# %% [markdown]
# Finally, before starting with our implementation of SimCLR, let's look
# at some example image pairs sampled with our augmentations:

# %%
# Visualize some examples
L.seed_everything(42)
NUM_IMAGES = 6
imgs = torch.stack([img for idx in range(NUM_IMAGES) for img in unlabeled_data[idx][0]], dim=0)
img_grid = torchvision.utils.make_grid(imgs, nrow=6, normalize=True, pad_value=0.9)
img_grid = img_grid.permute(1, 2, 0)

plt.figure(figsize=(10, 5))
plt.title("Augmented image examples of the STL10 dataset")
plt.imshow(img_grid)
plt.axis("off")
plt.show()
plt.close()

# %% [markdown]
# We see the wide variety of our data augmentation, including randomly cropping, grayscaling, gaussian blur, and color distortion.
# Thus, it remains a challenging task for the model to match two, independently augmented patches of the same image.

# %% [markdown]
# ### SimCLR implementation
#
# Using the data loader pipeline above, we can now implement SimCLR.
# At each iteration, we get for every image $x$ two differently augmented versions, which we refer to as $\tilde{x}_i$ and $\tilde{x}_j$.
# Both of these images are encoded into a one-dimensional feature vector, between which we want to maximize similarity which minimizes it to all other images in the batch.
# The encoder network is split into two parts: a base encoder network $f(\cdot)$, and a projection head $g(\cdot)$.
# The base network is usually a deep CNN as we have seen in e.g. [Tutorial 5](https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial5/Inception_ResNet_DenseNet.html) before, and is responsible for extracting a representation vector from the augmented data examples.
# In our experiments, we will use the common ResNet-18 architecture as $f(\cdot)$, and refer to the output as $f(\tilde{x}_i)=h_i$.
# The projection head $g(\cdot)$ maps the representation $h$ into a space where we apply the contrastive loss, i.e., compare similarities between vectors.
# It is often chosen to be a small MLP with non-linearities, and for simplicity, we follow the original SimCLR paper setup by defining it as a two-layer MLP with ReLU activation in the hidden layer.
# Note that in the follow-up paper, [SimCLRv2](https://arxiv.org/abs/2006.10029), the authors mention that larger/wider MLPs can boost the performance considerably.
# This is why we apply an MLP with four times larger hidden dimensions, but deeper MLPs showed to overfit on the given dataset.
# The general setup is visualized below (figure credit - [Ting Chen et al. ](https://arxiv.org/abs/2006.10029)):
#
# <center width="100%"><img src="simclr_network_setup.svg" width="350px"></center>
#
# After finishing the training with contrastive learning, we will remove the projection head $g(\cdot)$, and use $f(\cdot)$ as a pretrained feature extractor.
# The representations $z$ that come out of the projection head $g(\cdot)$ have been shown to perform worse than those of the base network $f(\cdot)$ when finetuning the network for a new task.
# This is likely because the representations $z$ are trained to become invariant to many features like the color that can be important for downstream tasks.
# Thus, $g(\cdot)$ is only needed for the contrastive learning stage.
#
# Now that the architecture is described, let's take a closer look at how we train the model.
# As mentioned before, we want to maximize the similarity between the representations of the two augmented versions of the same image, i.e., $z_i$ and $z_j$ in the figure above, while minimizing it to all other examples in the batch.
# SimCLR thereby applies the InfoNCE loss, originally proposed by [Aaron van den Oord et al. ](https://arxiv.org/abs/1807.03748) for contrastive learning.
# In short, the InfoNCE loss compares the similarity of $z_i$ and $z_j$ to the similarity of $z_i$ to any other representation in the batch by performing a softmax over the similarity values.
# The loss can be formally written as:
# $$
# \ell_{i,j}=-\log \frac{\exp(\text{sim}(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)}=-\text{sim}(z_i,z_j)/\tau+\log\left[\sum_{k=1}^{2N}\mathbb{1}_{[k\neq i]}\exp(\text{sim}(z_i,z_k)/\tau)\right]
# $$
# The function $\text{sim}$ is a similarity metric, and the hyperparameter $\tau$ is called temperature determining how peaked the distribution is.
# Since many similarity metrics are bounded, the temperature parameter allows us to balance the influence of many dissimilar image patches versus one similar patch.
# The similarity metric that is used in SimCLR is cosine similarity, as defined below:
# $$
# \text{sim}(z_i,z_j) = \frac{z_i^\top \cdot z_j}{||z_i||\cdot||z_j||}
# $$
# The maximum cosine similarity possible is $1$, while the minimum is $-1$.
# In general, we will see that the features of two different images will converge to a cosine similarity around zero since the minimum, $-1$, would require $z_i$ and $z_j$ to be in the exact opposite direction in all feature dimensions, which does not allow for great flexibility.
#
# Finally, now that we have discussed all details, let's implement SimCLR below as a PyTorch Lightning module:


# %%
class SimCLR(L.LightningModule):
    def __init__(self, hidden_dim, lr, temperature, weight_decay, max_epochs=500):
        super().__init__()
        self.save_hyperparameters()
        assert self.hparams.temperature > 0.0, "The temperature must be a positive float!"
        # Base model f(.)
        self.convnet = torchvision.models.resnet18(
            pretrained=False, num_classes=4 * hidden_dim
        )  # num_classes is the output size of the last linear layer
        # The MLP for g(.) consists of Linear->ReLU->Linear
        self.convnet.fc = nn.Sequential(
            self.convnet.fc,  # Linear(ResNet output, 4*hidden_dim)
            nn.ReLU(inplace=True),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.hparams.max_epochs, eta_min=self.hparams.lr / 50
        )
        return [optimizer], [lr_scheduler]

    def info_nce_loss(self, batch, mode="train"):
        imgs, _ = batch
        imgs = torch.cat(imgs, dim=0)

        # Encode all images
        feats = self.convnet(imgs)
        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)
        # Find positive example -> batch_size//2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)
        # InfoNCE loss
        cos_sim = cos_sim / self.hparams.temperature
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        # Logging loss
        self.log(mode + "_loss", nll)
        # Get ranking position of positive example
        comb_sim = torch.cat(
            [cos_sim[pos_mask][:, None], cos_sim.masked_fill(pos_mask, -9e15)],  # First position positive example
            dim=-1,
        )
        sim_argsort = comb_sim.argsort(dim=-1, descending=True).argmin(dim=-1)
        # Logging ranking metrics
        self.log(mode + "_acc_top1", (sim_argsort == 0).float().mean())
        self.log(mode + "_acc_top5", (sim_argsort < 5).float().mean())
        self.log(mode + "_acc_mean_pos", 1 + sim_argsort.float().mean())

        return nll

    def training_step(self, batch, batch_idx):
        return self.info_nce_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self.info_nce_loss(batch, mode="val")


# %% [markdown]
# Alternatively to performing the validation on the contrastive learning loss as well, we could also take a simple, small downstream task, and track the performance of the base network $f(\cdot)$ on that.
# However, in this tutorial, we will restrict ourselves to the STL10
# dataset where we use the task of image classification on STL10 as our
# test task.

# %% [markdown]
# ### Training
#
# Now that we have implemented SimCLR and the data loading pipeline, we are ready to train the model.
# We will use the same training function setup as usual.
# For saving the best model checkpoint, we track the metric `val_acc_top5`, which describes how often the correct image patch is within the top-5 most similar examples in the batch.
# This is usually less noisy than the top-1 metric, making it a better metric to choose the best model from.


# %%
def train_simclr(batch_size, max_epochs=500, **kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
        ],
    )
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        # Automatically loads the model with the saved hyperparameters
        model = SimCLR.load_from_checkpoint(pretrained_filename)
    else:
        train_loader = data.DataLoader(
            unlabeled_data,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = data.DataLoader(
            train_data_contrast,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        L.seed_everything(42)  # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        # Load best checkpoint after training
        model = SimCLR.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    return model


# %% [markdown]
# A common observation in contrastive learning is that the larger the batch size, the better the models perform.
# A larger batch size allows us to compare each image to more negative examples, leading to overall smoother loss gradients.
# However, in our case, we experienced that a batch size of 256 was sufficient to get good results.

# %%
simclr_model = train_simclr(
    batch_size=256, hidden_dim=128, lr=5e-4, temperature=0.07, weight_decay=1e-4, max_epochs=500
)

# %% [markdown]
# To get an intuition of how training with contrastive learning behaves, we can take a look at the TensorBoard below:

# %%
# %tensorboard --logdir ../saved_models/tutorial17/tensorboards/SimCLR/

# %% [markdown]
# <center width="100%"> ![tensorboard simclr](tensorboard_simclr.png){width="1200px"} </center>
#
# One thing to note is that contrastive learning benefits a lot from long training.
# The shown plot above is from a training that took approx.
# 1 day on a NVIDIA TitanRTX.
# Training the model for even longer might reduce its loss further, but we did not experience any gains from it for the downstream task on image classification.
# In general, contrastive learning can also benefit from using larger models, if sufficient unlabeled data is available.

# %% [markdown]
# ## Logistic Regression
#
# <div class="center-wrapper"><div class="video-wrapper"><iframe src="https://www.youtube.com/embed/o3FktysLLd4" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe></div></div>
# After we have trained our model via contrastive learning, we can deploy it on downstream tasks and see how well it performs with little data.
# A common setup, which also verifies whether the model has learned generalized representations, is to perform Logistic Regression on the features.
# In other words, we learn a single, linear layer that maps the representations to a class prediction.
# Since the base network $f(\cdot)$ is not changed during the training process, the model can only perform well if the representations of $h$ describe all features that might be necessary for the task.
# Further, we do not have to worry too much about overfitting since we have very few parameters that are trained.
# Hence, we might expect that the model can perform well even with very little data.
#
# First, let's implement a simple Logistic Regression setup for which we assume that the images already have been encoded in their feature vectors.
# If very little data is available, it might be beneficial to dynamically encode the images during training so that we can also apply data augmentations.
# However, the way we implement it here is much more efficient and can be trained within a few seconds.
# Further, using data augmentations did not show any significant gain in this simple setup.


# %%
class LogisticRegression(L.LightningModule):
    def __init__(self, feature_dim, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        # Mapping from representation h to classes
        self.model = nn.Linear(feature_dim, num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.hparams.max_epochs * 0.6), int(self.hparams.max_epochs * 0.8)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        feats, labels = batch
        preds = self.model(feats)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


# %% [markdown]
# The data we use is the training and test set of STL10.
# The training contains 500 images per class, while the test set has 800 images per class.

# %%
img_transforms = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_img_data = STL10(root=DATASET_PATH, split="train", download=True, transform=img_transforms)
test_img_data = STL10(root=DATASET_PATH, split="test", download=True, transform=img_transforms)

print("Number of training examples:", len(train_img_data))
print("Number of test examples:", len(test_img_data))

# %% [markdown]
# Next, we implement a small function to encode all images in our datasets.
# The output representations are then used as inputs to the Logistic Regression model.


# %%
@torch.no_grad()
def prepare_data_features(model, dataset):
    # Prepare model
    network = deepcopy(model.convnet)
    network.fc = nn.Identity()  # Removing projection head g(.)
    network.eval()
    network.to(device)

    # Encode all images
    data_loader = data.DataLoader(dataset, batch_size=64, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    feats, labels = [], []
    for batch_imgs, batch_labels in tqdm(data_loader):
        batch_imgs = batch_imgs.to(device)
        batch_feats = network(batch_imgs)
        feats.append(batch_feats.detach().cpu())
        labels.append(batch_labels)

    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)

    # Sort images by labels
    labels, idxs = labels.sort()
    feats = feats[idxs]

    return data.TensorDataset(feats, labels)


# %% [markdown]
# Let's apply the function to both training and test set below.

# %%
train_feats_simclr = prepare_data_features(simclr_model, train_img_data)
test_feats_simclr = prepare_data_features(simclr_model, test_img_data)

# %% [markdown]
# Finally, we can write a training function as usual.
# We evaluate the model on the test set every 10 epochs to allow early
# stopping, but the low frequency of the validation ensures that we do not
# overfit too much on the test set.


# %%
def train_logreg(batch_size, train_feats_data, test_feats_data, model_suffix, max_epochs=100, **kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "LogisticRegression"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        enable_progress_bar=False,
        check_val_every_n_epoch=10,
    )
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(
        train_feats_data, batch_size=batch_size, shuffle=True, drop_last=False, pin_memory=True, num_workers=0
    )
    test_loader = data.DataLoader(
        test_feats_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=0
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, f"LogisticRegression_{model_suffix}.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = LogisticRegression.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = LogisticRegression(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = LogisticRegression.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on train and validation set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": test_result[0]["test_acc"]}

    return model, result


# %% [markdown]
# Despite the training dataset of STL10 already only having 500 labeled images per class, we will perform experiments with even smaller datasets.
# Specifically, we train a Logistic Regression model for datasets with only 10, 20, 50, 100, 200, and all 500 examples per class.
# This gives us an intuition on how well the representations learned by contrastive learning can be transfered to a image recognition task like this classification.
# First, let's define a function to create the intended sub-datasets from the full training set:


# %%
def get_smaller_dataset(original_dataset, num_imgs_per_label):
    new_dataset = data.TensorDataset(
        *(t.unflatten(0, (10, 500))[:, :num_imgs_per_label].flatten(0, 1) for t in original_dataset.tensors)
    )
    return new_dataset


# %% [markdown]
# Next, let's run all models.
# Despite us training 6 models, this cell could be run within a minute or two without the pretrained models.

# %%
results = {}
for num_imgs_per_label in [10, 20, 50, 100, 200, 500]:
    sub_train_set = get_smaller_dataset(train_feats_simclr, num_imgs_per_label)
    _, small_set_results = train_logreg(
        batch_size=64,
        train_feats_data=sub_train_set,
        test_feats_data=test_feats_simclr,
        model_suffix=num_imgs_per_label,
        feature_dim=train_feats_simclr.tensors[0].shape[1],
        num_classes=10,
        lr=1e-3,
        weight_decay=1e-3,
    )
    results[num_imgs_per_label] = small_set_results

# %% [markdown]
# Finally, let's plot the results.

# %%
dataset_sizes = sorted(k for k in results)
test_scores = [results[k]["test"] for k in dataset_sizes]

fig = plt.figure(figsize=(6, 4))
plt.plot(
    dataset_sizes,
    test_scores,
    "--",
    color="#000",
    marker="*",
    markeredgecolor="#000",
    markerfacecolor="y",
    markersize=16,
)
plt.xscale("log")
plt.xticks(dataset_sizes, labels=dataset_sizes)
plt.title("STL10 classification over dataset size", fontsize=14)
plt.xlabel("Number of images per class")
plt.ylabel("Test accuracy")
plt.minorticks_off()
plt.show()

for k, score in zip(dataset_sizes, test_scores):
    print(f"Test accuracy for {k:3d} images per label: {100*score:4.2f}%")

# %% [markdown]
# As one would expect, the classification performance improves the more data we have.
# However, with only 10 images per class, we can already classify more than 60% of the images correctly.
# This is quite impressive, considering that the images are also higher dimensional than e.g. CIFAR10.
# With the full dataset, we achieve an accuracy of 81%.
# The increase between 50 to 500 images per class might suggest a linear increase in performance with an exponentially larger dataset.
# However, with even more data, we could also finetune $f(\cdot)$ in the training process, allowing for the representations to adapt more to the specific classification task given.
#
# To set the results above into perspective, we will train the base
# network, a ResNet-18, on the classification task from scratch.

# %% [markdown]
# ## Baseline
#
# As a baseline to our results above, we will train a standard ResNet-18 with random initialization on the labeled training set of STL10.
# The results will give us an indication of the advantages that contrastive learning on unlabeled data has compared to using only supervised training.
# The implementation of the model is straightforward since the ResNet
# architecture is provided in the torchvision library.


# %%
class ResNet(L.LightningModule):
    def __init__(self, num_classes, lr, weight_decay, max_epochs=100):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[int(self.hparams.max_epochs * 0.7), int(self.hparams.max_epochs * 0.9)], gamma=0.1
        )
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode="train"):
        imgs, labels = batch
        preds = self.model(imgs)
        loss = F.cross_entropy(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        self.log(mode + "_loss", loss)
        self.log(mode + "_acc", acc)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="test")


# %% [markdown]
# It is clear that the ResNet easily overfits on the training data since its parameter count is more than 1000 times larger than the dataset size.
# To make the comparison to the contrastive learning models fair, we apply data augmentations similar to the ones we used before: horizontal flip, crop-and-resize, grayscale, and gaussian blur.
# Color distortions as before are not used because the color distribution of an image showed to be an important feature for the classification.
# Hence, we observed no noticeable performance gains when adding color distortions to the set of augmentations.
# Similarly, we restrict the resizing operation before cropping to the max.
# 125% of its original resolution, instead of 1250% as done in SimCLR.
# This is because, for classification, the model needs to recognize the full object, while in contrastive learning, we only want to check whether two patches belong to the same image/object.
# Hence, the chosen augmentations below are overall weaker than in the contrastive learning case.

# %%
train_transforms = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=96, scale=(0.8, 1.0)),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=9, sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
)

train_img_aug_data = STL10(root=DATASET_PATH, split="train", download=True, transform=train_transforms)

# %% [markdown]
# The training function for the ResNet is almost identical to the Logistic Regression setup.
# Note that we allow the ResNet to perform validation every 2 epochs to
# also check whether the model overfits strongly in the first iterations
# or not.


# %%
def train_resnet(batch_size, max_epochs=100, **kwargs):
    trainer = L.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "ResNet"),
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc"),
            LearningRateMonitor("epoch"),
        ],
        check_val_every_n_epoch=2,
    )
    trainer.logger._default_hp_metric = None

    # Data loaders
    train_loader = data.DataLoader(
        train_img_aug_data,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = data.DataLoader(
        test_img_data, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=True, num_workers=NUM_WORKERS
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "ResNet.ckpt")
    if os.path.isfile(pretrained_filename):
        print("Found pretrained model at %s, loading..." % pretrained_filename)
        model = ResNet.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)  # To be reproducable
        model = ResNet(**kwargs)
        trainer.fit(model, train_loader, test_loader)
        model = ResNet.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

    # Test best model on validation set
    train_result = trainer.test(model, dataloaders=train_loader, verbose=False)
    val_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"train": train_result[0]["test_acc"], "test": val_result[0]["test_acc"]}

    return model, result


# %% [markdown]
# Finally, let's train the model and check its results:

# %%
resnet_model, resnet_result = train_resnet(batch_size=64, num_classes=10, lr=1e-3, weight_decay=2e-4, max_epochs=100)
print(f"Accuracy on training set: {100*resnet_result['train']:4.2f}%")
print(f"Accuracy on test set: {100*resnet_result['test']:4.2f}%")

# %% [markdown]
# The ResNet trained from scratch achieves 73.31% on the test set.
# This is almost 8% less than the contrastive learning model, and even slightly less than SimCLR achieves with 1/10 of the data.
# This shows that self-supervised, contrastive learning provides
# considerable performance gains by leveraging large amounts of unlabeled
# data when little labeled data is available.

# %% [markdown]
# ## Conclusion
#
# In this tutorial, we have discussed self-supervised contrastive learning and implemented SimCLR as an example method.
# We have applied it to the STL10 dataset and showed that it can learn generalizable representations that we can use to train simple classification models.
# With 500 images per label, it achieved an 8% higher accuracy than a similar model solely trained from supervision and performs on par with it when only using a tenth of the labeled data.
# Our experimental results are limited to a single dataset, but recent works such as [Ting Chen et al. ](https://arxiv.org/abs/2006.10029) showed similar trends for larger datasets like ImageNet.
# Besides the discussed hyperparameters, the size of the model seems to be important in contrastive learning as well.
# If a lot of unlabeled data is available, larger models can achieve much stronger results and come close to their supervised baselines.
# Further, there are also approaches for combining contrastive and supervised learning, leading to performance gains beyond supervision (see [Khosla et al.](https://arxiv.org/abs/2004.11362)).
# Moreover, contrastive learning is not the only approach to self-supervised learning that has come up in the last two years and showed great results.
# Other methods include distillation-based methods like [BYOL](https://arxiv.org/abs/2006.07733) and redundancy reduction techniques like [Barlow Twins](https://arxiv.org/abs/2103.03230).
# There is a lot more to explore in the self-supervised domain, and more, impressive steps ahead are to be expected.
#
# ### References
#
# [1] Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. (2020).
# A simple framework for contrastive learning of visual representations.
# In International conference on machine learning (pp.
# 1597-1607).
# PMLR.
# ([link](https://arxiv.org/abs/2002.05709))
#
# [2] Chen, T., Kornblith, S., Swersky, K., Norouzi, M., and Hinton, G. (2020).
# Big self-supervised models are strong semi-supervised learners.
# NeurIPS 2021 ([link](https://arxiv.org/abs/2006.10029)).
#
# [3] Oord, A. V. D., Li, Y., and Vinyals, O.
# (2018).
# Representation learning with contrastive predictive coding.
# arXiv preprint arXiv:1807.03748.
# ([link](https://arxiv.org/abs/1807.03748))
#
# [4] Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G.
# and Piot, B.
# (2020).
# Bootstrap your own latent: A new approach to self-supervised learning.
# arXiv preprint arXiv:2006.07733.
# ([link](https://arxiv.org/abs/2006.07733))
#
# [5] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C. and Krishnan, D. (2020).
# Supervised contrastive learning.
# arXiv preprint arXiv:2004.11362.
# ([link](https://arxiv.org/abs/2004.11362))
#
# [6] Zbontar, J., Jing, L., Misra, I., LeCun, Y. and Deny, S. (2021).
# Barlow twins: Self-supervised learning via redundancy reduction.
# arXiv preprint arXiv:2103.03230.
# ([link](https://arxiv.org/abs/2103.03230))
