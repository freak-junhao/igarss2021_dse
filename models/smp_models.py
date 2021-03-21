from abc import ABC

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DownsizeBlock(nn.Module, ABC):

    def __init__(self, out_channels, downsize_mode=2):
        super(DownsizeBlock, self).__init__()

        if downsize_mode == 0:
            self.downsize = nn.Sequential(
                nn.Conv2d(
                    in_channels=out_channels, out_channels=out_channels, kernel_size=50, stride=50
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        elif downsize_mode == 1:
            self.downsize = nn.AdaptiveAvgPool2d((16, 16))
        else:
            self.downsize = nn.MaxPool2d((50, 50))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.downsize(x)


class SmpModel16(nn.Module, ABC):

    def __init__(self, model_name, in_channels=3, out_channels=1):
        super(SmpModel16, self).__init__()

        aux_params = dict(
            pooling='max',  # one of 'avg', 'max'
            dropout=0.5,  # dropout ratio, default is None
            activation='softmax',  # activation function, default is None
            classes=out_channels,  # define number of output labels
        )

        if 'unet' in model_name:
            self.model = smp.Unet(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_depth=3,
                encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
                decoder_channels=[128, 64, 32],  # [256, 128, 64, 32]
                in_channels=in_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channels,  # model output channels (number of classes in your dataset)
                aux_params=aux_params,
            )

        elif 'uplus' in model_name:

            self.model = smp.UnetPlusPlus(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_depth=3,
                encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
                decoder_channels=[256, 128, 64],
                in_channels=in_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channels,  # model output channels (number of classes in your dataset)
                aux_params=aux_params,
            )

        else:

            self.model = smp.PSPNet(
                encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                # encoder_depth=4,
                encoder_weights="imagenet",  # use `imagenet` pretrained weights for encoder initialization
                # decoder_channels=[256, 128, 64, 32],
                in_channels=in_channels,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
                classes=out_channels,  # model output channels (number of classes in your dataset)
                aux_params=aux_params,
            )

        self.down_layer = DownsizeBlock(out_channels, downsize_mode=2)

    def forward(self, x):
        x, _ = self.model(x)
        x = self.down_layer(x)

        return x


if __name__ == '__main__':
    model = SmpModel16('unet', in_channels=3, out_channels=1)
    print(model)

    model_input = torch.randn(1, 3, 800, 800)
    out = model(model_input)
    print(out.shape)
