import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything, Trainer
from tutorial_model import MNISTDataModule, TutorialModule


class DDPAllGatherDemoModule(TutorialModule):

    def test_step(self, batch, batch_idx):
        x, y = batch
        prob = F.softmax(self(x), dim=1)
        pred = torch.argmax(prob, dim=1)
        return pred

    def test_epoch_end(self, outputs):
        process_id = self.global_rank
        preds = torch.cat(outputs)
        print(f"{process_id=} saw {len(outputs)} test_step outputs, made {len(preds)} predictions")

        # gather all predictions into all processes
        all_preds = self.all_gather(preds)
        print(f"{process_id=} all-gathered {all_preds.shape[0]} x {all_preds.shape[1]} predictions")

        # do something will all outputs
        # ...


if __name__ == "__main__":
    seed_everything(1)
    model = DDPAllGatherDemoModule()
    datamodule = MNISTDataModule()

    trainer = Trainer(
        gpus=4,
        accelerator="ddp",
        max_epochs=1,
    )
    trainer.test(model, datamodule=datamodule)
