import torch
import pytorch_lightning as pl


class Seq2SeqTransformer(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()

    def forward(self, input_tensor: torch.Tensor):
        # TODO: implement forward method body
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        # TODO: implement train_step method body
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        # TODO: implement validation_step body method
        raise NotImplementedError()

    def configure_optimizers(self):
        # TODO: implement configure_optimizers method body
        raise NotImplementedError()
