import pytorch_lightning as pl
import torch


class BaseModel(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.learning_rate = args.learning_rate

    def forward(self):
        pass

    def training_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        # logs
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        self.log('validation_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_each_epoch', avg_loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        # batch
        src_ids, decoder_ids, mask, label_ids = batch
        # get loss
        loss = self(input_ids=src_ids, attention_mask=mask, decoder_input_ids=decoder_ids, labels=label_ids)
        return loss

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('test_loss', avg_loss, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.scheduler_lambda1,
                                                    gamma=self.args.scheduler_lambda2, verbose=True)
        return [optimizer], [scheduler]
