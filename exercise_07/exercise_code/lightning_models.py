
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision

from torchvision import transforms
from torch.utils.data import DataLoader, random_split


class TwoLayerNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # This sets self.hparams the the dict or namespace
        self.save_hyperparameters(hparams)

        # We can access the parameters here
        self.model = nn.Sequential(
            nn.Linear(self.hparams.input_size,
                      self.hparams.hidden_size),
            nn.Sigmoid(),
            nn.Linear(self.hparams.hidden_size,
                      self.hparams.num_classes),
        )

    def forward(self, x):
        # flatten the image  before sending as input to the model
        N, _, _, _ = x.shape
        x = x.view(N, -1)

        x = self.model(x)

        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Log the accuracy and loss values to the tensorboard
        self.log('loss', loss)
        
        # Or also on the progress bar in our console/notebook
        # if you want it to not show in tensorboard just disable
        # the logger but usually you want to log everything :)
        self.log('acc', acc, logger=True, prog_bar=True)

        # Ultimately we return the loss which will be then used
        # for backpropagation in pytorch lightning automatically
        # This will always be logged in the progressbar as well
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # Perform a forward pass on the network with inputs
        out = self.forward(images)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.cross_entropy(out, targets)

        # Find the predicted class from probabilites of the image belonging to each of the classes
        # from the network output
        _, preds = torch.max(out, 1)

        # Calculate the accuracy of predictions
        acc = preds.eq(targets).sum().float() / targets.size(0)

        # Visualise the predictions  of the model each epoch
        if batch_idx == 0:
            self.visualize_predictions(images.detach(), out.detach(), targets)

        # Whatever we return here, we have access to in the 
        # validation epoch end function. A dictionary is more
        # ordered than a list or tuple
        self.log("val_loss", loss, logger=True, prog_bar=True)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):

        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.model.parameters(
        ), self.hparams["learning_rate"], momentum=0.9)

        return optim

    def visualize_predictions(self, images, preds, targets):

        # Helper function to help us visualize the predictions of the
        # validation data by the model

        class_names = ['t-shirts', 'trouser', 'pullover', 'dress',
                       'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

        # determine size of the grid based for the given batch size
        num_rows = int(torch.tensor(len(images)).float().sqrt().floor())

        fig = plt.figure(figsize=(10, 10))
        for i in range(len(images)):
            plt.subplot(num_rows, len(images) // num_rows + 1, i+1)
            image = images[i].cpu().numpy().squeeze(0)
            image = image / 2 + 0.5     # unnormalize
            plt.imshow(image, cmap="Greys")
            plt.title(class_names[torch.argmax(preds, axis=-1)
                                  [i]] + f'\n[{class_names[targets[i]]}]')
            plt.axis('off')

        self.logger.experiment.add_figure(
            'predictions', fig, global_step=self.global_step)
