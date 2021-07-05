import torch
from pytorch_lightning import seed_everything, Trainer
from tutorial_model import MNISTDataModule, TutorialModule

if __name__ == "__main__":
    seed_everything(1)
    model = TutorialModule()
    datamodule = MNISTDataModule()

    # trainer = Trainer(
    #     gpus=2,
    #     accelerator="ddp",
    #     max_epochs=1,
    # )
    # trainer.fit(model, datamodule=datamodule)
    #
    # ddp_max_mem = torch.cuda.max_memory_allocated(trainer.local_rank) / 1000
    # print(f"GPU {trainer.local_rank} max memory using DDP: {ddp_max_mem:.2f} MB")

    trainer = Trainer(
        gpus=2,
        accelerator="ddp_sharded",
        max_epochs=1,
    )
    trainer.fit(model, datamodule=datamodule)

    sdp_max_mem = torch.cuda.max_memory_allocated(trainer.local_rank) / 1000
    print(f"GPU {trainer.local_rank} max memory using SDP: {sdp_max_mem:.2f} MB")
