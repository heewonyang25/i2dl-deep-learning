"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models

#class SegmentationNN(pl.LightningModule):

class SegmentationNN(nn.Module):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        #self.save_hyperparameters(hparams)
        ########################################################################
        # TODO - Train Your Model                                              #
        ########################################################################
        
        num_classes = num_classes+1
        
        alexnet = models.alexnet(pretrained=True)
        '''
        Sequential(
  (0): Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
  (1): ReLU(inplace=True)
  (2): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
  (4): ReLU(inplace=True)
  (5): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (7): ReLU(inplace=True)
  (8): Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (9): ReLU(inplace=True)
  (10): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)
)'''
        
        # Convolution layers for feature extraction
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=21, stride=4, padding=2),  # conv2d(240x240x3, 55x55x64, kernel=21x21, stride=4, padding=2)
            alexnet.features[1],  # relu
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False),  # maxpool2d(55x55x64, 27x27x64, kernel 3x3, stride=2, padding=0) pool 1
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),  # conv2d(27x27x64, 27x27x192, kernel=3x3, stride=1, padding=1)
            alexnet.features[4],  # relu
            nn.MaxPool2d(kernel_size=5, stride=2, padding=1, dilation=1, ceil_mode=False),  # maxpool2d(27x27x192, 13x13x192, kernel=3x3, stride=2, padding=1) pool 2
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # conv2d(13x13x192, 13x13x384, kernel=3x3, stride=1, padding=1)
            alexnet.features[7],  # relu
        )
        self.conv4 = nn.Sequential(
            alexnet.features[10],  # conv2d(13x13x256, 13x13x256, kernel=3x3, stride=1, padding=1)
            alexnet.features[11],  # relu
        )
        self.conv5 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, dilation=1, ceil_mode=False)  # maxpool2d(13x13x256, 6x6x256, kernel=5x5, stride=2) pool 3

        # size-1 convolution for pixel-by-pixel prediction
        self.score_conv = nn.Conv2d(256, num_classes, 1)

        # Deconvolution layers for restoring the original image
        self.deconv1 = nn.ConvTranspose2d(num_classes, 256, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2)
        self.deconv3 = nn.ConvTranspose2d(64, 192, kernel_size=3, stride=2)
        self.deconv4 = nn.ConvTranspose2d(192, num_classes, kernel_size=24, stride=4)

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

        print('input shape', x.size())
        
        out_conv1 = self.conv1(x)
        print('conv 1 output shape', out_conv1.size())

        out_conv2 = self.conv2(out_conv1)
        print('conv 2 output shape', out_conv2.size())

        out_conv3 = self.conv3(out_conv2)
        print('conv 3 output shape', out_conv3.size())

        out_conv4 = self.conv4(out_conv3)
        print('conv 4 output shape', out_conv4.size())

        out_conv5 = self.conv5(out_conv4)
        print('conv 5 output shape', out_conv5.size())

        out_score_conv = self.score_conv(out_conv5)
        print('score conv output shape', out_score_conv.size())

        out_deconv1 = self.deconv1(out_score_conv)
        print('deconv 1 output shape', out_deconv1.size())

        out_deconv2 = self.deconv2(out_deconv1 + out_conv4)
        print('deconv 2 output shape', out_deconv2.size())

        out_deconv3 = self.deconv3(out_deconv2 + out_conv1)
        print('deconv 3 output shape', out_deconv3.size())

        out_deconv4 = self.deconv4(out_deconv3)
        print('deconv 4 output shape', out_deconv4.size())
        
        x = out_deconv4

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
