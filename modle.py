import argparse
import os
import h5py
import torch
from copy import deepcopy
import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.optim.lr_scheduler
import torch.utils.data.distributed
import torchvision.transforms as transforms
from transformers import AdamW
from torch.cuda.amp import GradScaler, autocast
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import _LRScheduler
import torch
import math
import timm
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy, precision
import torchmetrics.functional as tf
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks import StochasticWeightAveraging
from positional_encodings.torch_encodings import PositionalEncodingPermute2D, Summer


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=(11, 5), stride=2, padding=(5, 0), bias=False)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        return x


class Query2Label(nn.Module):
    """Modified Query2Label model

    Unlike the model described in the paper (which uses a modified DETR
    transformer), this version uses a standard, unmodified Pytorch Transformer.
    Learnable label embeddings are passed to the decoder module as the target
    sequence (and ultimately is passed as the Query to MHA).
    """

    def __init__(
            self, conv_out, num_classes, hidden_dim=256, nheads=8,
            encoder_layers=6, decoder_layers=6, use_pos_encoding=False):
        """Initializes model

        Args:
            model (str): Timm model descriptor for backbone.
            conv_out (int): Backbone output channels.
            num_classes (int): Number of possible label classes
            hidden_dim (int, optional): Hidden channels from linear projection of
            backbone output. Defaults to 256.
            nheads (int, optional): Number of MHA heads. Defaults to 8.
            encoder_layers (int, optional): Number of encoders. Defaults to 6.
            decoder_layers (int, optional): Number of decoders. Defaults to 6.
            use_pos_encoding (bool, optional): Flag for use of position encoding.
            Defaults to False.
        """

        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.use_pos_encoding = use_pos_encoding

        self.backbone = ResNet50()
        self.conv = nn.Conv2d(conv_out, hidden_dim, 1)
        self.transformer = nn.Transformer(
            hidden_dim, nheads, encoder_layers, decoder_layers)

        if self.use_pos_encoding:
            # returns the encoding object
            self.pos_encoder = PositionalEncodingPermute2D(hidden_dim)

            # returns the summing object
            self.encoding_adder = Summer(self.pos_encoder)

        # prediction head
        self.classifier = nn.Linear(num_classes * hidden_dim, num_classes)

        # learnable label embedding
        self.label_emb = nn.Parameter(torch.rand(1, num_classes, hidden_dim))

    def forward(self, x):
        """Passes batch through network

        Args:
            x (Tensor): Batch of images

        Returns:
            Tensor: Output of classification head
        """
        # produces output of shape [N x C x H x W]
        out = self.backbone(x)

        # reduce number of feature planes for the transformer
        h = self.conv(out)
        B, C, H, W = h.shape

        # add position encodings
        if self.use_pos_encoding:
            # input with encoding added
            h = self.encoding_adder(h * 0.1)

        # convert h from [N x C x H x W] to [H*W x N x C] (N=batch size)
        # this corresponds to the [SIZE x BATCH_SIZE x EMBED_DIM] dimensions
        # that the transformer expects
        h = h.flatten(2).permute(2, 0, 1)

        # image feature vector "h" is sent in after transformation above; we
        # also convert label_emb from [1 x TARGET x (hidden)EMBED_SIZE] to
        # [TARGET x BATCH_SIZE x (hidden)EMBED_SIZE]
        label_emb = self.label_emb.repeat(B, 1, 1)
        label_emb = label_emb.transpose(0, 1)
        h = self.transformer(h, label_emb).transpose(0, 1)

        # output from transformer was of dim [TARGET x BATCH_SIZE x EMBED_SIZE];
        # however, we transposed it to [BATCH_SIZE x TARGET x EMBED_SIZE] above.
        # below we reshape to [BATCH_SIZE x TARGET*EMBED_SIZE].
        #
        # next, we project transformer outputs to class labels
        h = torch.reshape(h, (B, self.num_classes * self.hidden_dim))

        return self.classifier(h)


import torch
import torch.nn as nn
from utils_project.self_attention import NonLocalBlockND

class SAResnetModel(nn.Module):
    def __init__(self, config, kernal_channel, fc_num):
        super(SAResnetModel, self).__init__()
        self.config = config
        self.k_c = kernal_channel
        self.fc_num = fc_num
        self.channels = 4
        self.height = 1
        self.weight = 101
        self.dropout_rate = 0.7
        self.build_model()

    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2 = out_filters

        X_shortcut = X_input

        # first
        X = nn.BatchNorm2d(in_filter)(X_input)
        X = nn.ELU()(X)
        W_conv1 = nn.Parameter(torch.Tensor(1, kernel_size, in_filter, f1))
        nn.init.kaiming_normal_(W_conv1)
        X = nn.Conv2d(in_filter, f1, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size//2))(X)

        # second
        X = nn.BatchNorm2d(f1)(X)
        X = nn.ELU()(X)
        W_conv2 = nn.Parameter(torch.Tensor(1, kernel_size, f1, f2))
        nn.init.kaiming_normal_(W_conv2)
        X = nn.Conv2d(f1, f2, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size//2))(X)

        # final step
        add_result = X + X_shortcut

        return add_result

    def convolutional_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride=2):
        # defining name basis
        block_name = 'res' + str(stage) + block
        f1, f2 = out_filters

        x_shortcut = X_input

        # first
        X = nn.BatchNorm2d(in_filter)(X_input)
        X = nn.ELU()(X)
        W_conv1 = nn.Parameter(torch.Tensor(1, kernel_size, in_filter, f1))
        nn.init.kaiming_normal_(W_conv1)
        X = nn.Conv2d(in_filter, f1, kernel_size=(1, kernel_size), stride=(1, stride), padding=(0, kernel_size//2))(X)

        # second
        X = nn.BatchNorm2d(f1)(X)
        X = nn.ELU()(X)
        W_conv2 = nn.Parameter(torch.Tensor(1, kernel_size, f1, f2))
        nn.init.kaiming_normal_(W_conv2)
        X = nn.Conv2d(f1, f2, kernel_size=(1, kernel_size), stride=(1, 1), padding=(0, kernel_size//2))(X)

        # shortcut path
        W_shortcut = nn.Parameter(torch.Tensor(1, 1, in_filter, f2))
        nn.init.kaiming_normal_(W_shortcut)
        x_shortcut = nn.Conv2d(in_filter, f2, kernel_size=(1, 1), stride=(1, stride))(x_shortcut)

        # final
        add_result = x_shortcut + X

        return add_result

    def build_model(self):
        # network architecture
        self.w_conv1 = nn.Parameter(torch.Tensor(1, 7, self.channels, self.k_c))
        nn.init.kaiming_normal_(self.w_conv1)

        self.max_pooling = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

        self.non_local_1 = NonLocalBlockND(self.k_c, sub_sample=False, block='1')

        self.conv_block_2a = self.convolutional_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 2, 'a')
        self.identity_block_2b = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 2, 'b')
        self.identity_block_2c = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 2, 'c')
        self.non_local_2 = NonLocalBlockND(self.k_c, sub_sample=False, block='2')

        self.conv_block_3a = self.convolutional_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 3, 'a')
        self.identity_block_3b = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 3, 'b')
        self.identity_block_3c = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 3, 'c')
        self.identity_block_3d = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 3, 'd')
        self.non_local_3 = NonLocalBlockND(self.k_c, sub_sample=False, block='3')

        self.conv_block_4a = self.convolutional_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'a')
        self.identity_block_4b = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'b')
        self.identity_block_4c = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'c')
        self.identity_block_4d = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'd')
        self.identity_block_4e = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'e')
        self.identity_block_4f = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 4, 'f')

        self.conv_block_5a = self.convolutional_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 5, 'a')
        self.identity_block_5b = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 5, 'b')
        self.identity_block_5c = self.identity_block(self.k_c, 3, self.k_c, [self.k_c, self.k_c], 5, 'c')

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 7), stride=(1, 1))
        self.flatten = nn.Flatten()

        self.dropout_1 = nn.Dropout(p=self.dropout_rate)
        self.fc_1 = nn.Linear(self.k_c, self.fc_num)
        self.dropout_2 = nn.Dropout(p=self.dropout_rate)
        self.fc_2 = nn.Linear(self.fc_num, 2)

    def forward(self, x):
        x = nn.Conv2d(1, self.channels, kernel_size=(11, 5), stride=(2, 2), padding=(5, 0), bias=False)(x)

        x = nn.BatchNorm2d(self.channels)(x)
        x = nn.ELU()(x)

        x = nn.Conv2d(self.channels, self.k_c, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3), bias=False)(x)
        x = nn.ELU()(x)
        x = self.max_pooling(x)

        x = self.non_local_1(x)

        x = self.conv_block_2a(x)
        x = self.identity_block_2b(x)
        x = self.identity_block_2c(x)
        x = self.non_local_2(x)

        x = self.conv_block_3a(x)
        x = self.identity_block_3b(x)
        x = self.identity_block_3c(x)
        x = self.identity_block_3d(x)
        x = self.non_local_3(x)

        x = self.conv_block_4a(x)
        x = self.identity_block_4b(x)
        x = self.identity_block_4c(x)
        x = self.identity_block_4d(x)
        x = self.identity_block_4e(x)
        x = self.identity_block_4f(x)

        x = self.conv_block_5a(x)
        x = self.identity_block_5b(x)
        x = self.identity_block_5c(x)

        x = self.avg_pool(x)
        x = self.flatten(x)

        x = self.dropout_1(x)
        x = self.fc_1(x)
        x = self.dropout_2(x)
        logits = self.fc_2(x)

        return logits