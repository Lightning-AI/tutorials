from pytorch_lightning import seed_everything, Trainer
from tutorial_model import MNISTDataModule, TutorialModule


class DDPDataDemoModule(TutorialModule):

    def training_epoch_end(self, outputs):
        process_id = self.global_rank
        print(f"{process_id=} saw {len(outputs)} samples total")

    def on_train_end(self):
        print(f"training set contains {len(self.trainer.datamodule.mnist_train)} samples")


if __name__ == "__main__":
    seed_everything(1)
    model = DDPDataDemoModule()
    datamodule = MNISTDataModule(batch_size=1)

    trainer = Trainer(
        gpus=3,
        accelerator="ddp",
        max_epochs=1,
    )
    trainer.fit(model, datamodule=datamodule)
