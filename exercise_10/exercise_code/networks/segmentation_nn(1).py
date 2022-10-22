"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import torch.nn.functional as F

class SegmentationNN1(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        

        '''self.dconv_down1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(64, 64, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(32, 64)
        )
        self.dconv_down2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(128, 128, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(64, 128)
        )

        self.dconv_down3 = nn.Sequential(
            nn.Conv2d(3, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(128, 256)
        )
        self.dconv_down4 = nn.Sequential(
            nn.Conv2d(3, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(512, 512, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(256, 512)
        )        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.dconv_up3 = nn.Sequential(
            nn.Conv2d(768, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(256, 256, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(256 + 512, 256)
        )

        self.dconv_up2 = nn.Sequential(  
            nn.Conv2d(384, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(128, 128, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(128 + 256, 128)
        )
        self.dconv_up1 = nn.Sequential(
            nn.Conv2d(192, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            #nn.Conv2d(64, 64, 3, padding=1),
            #nn.ReLU(inplace=True)
            #double_conv(128 + 64, 64)
        )
        
        self.conv_last = nn.Conv2d(64, num_classes, 1)'''
        
        # convolutional layer
        self.cnn = nn.Sequential(
            nn.Conv2d(1, self.hparams["num_filters"], kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.hparams["num_filters"]*1, self.hparams["num_filters"]*2, kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.hparams["num_filters"]*2, self.hparams["num_filters"]*4, kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(p=0.2),
            nn.Conv2d(self.hparams["num_filters"]*4, self.hparams["num_filters"]*8, kernel_size=self.hparams["kernel_size"], padding=self.hparams["padding"]),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=0.2),
            nn.ReLU()
        )
        
        # fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(256*6*6, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 30),
            nn.Tanh()
        )
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        '''conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   
        
        x = self.dconv_down4(x)
        
        x = self.upsample(x)        
        x = torch.cat([x, conv3], dim=1)
        
        x = self.dconv_up3(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv2], dim=1)       

        x = self.dconv_up2(x)
        x = self.upsample(x)        
        x = torch.cat([x, conv1], dim=1)   
        
        x = self.dconv_up1(x)
        
        x = self.upsample(x)
        
        out = self.conv_last(x)'''
        
        x = self.cnn(x)
        x = x.view(-1, 256 * 6 * 6) # reshape tensor to (?, 256*6*6)
        x = self.fc(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()

        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
        