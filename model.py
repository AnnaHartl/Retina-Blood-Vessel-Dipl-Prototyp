import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU()

    def forward(self, x_conv):
        x_conv = self.conv1(x_conv)
        x_conv = self.bn1(x_conv)
        x_conv = self.relu(x_conv)

        x_conv = self.conv2(x_conv)
        x_conv = self.bn2(x_conv)
        x_conv = self.relu(x_conv)

        return x_conv


class EncoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv = ConvBlock(in_channel, out_channel)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, x_encoder):
        x_encoder = self.conv(x_encoder)
        p = self.pool(x_encoder)

        return x_encoder, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_channel + out_channel, out_channel)

    def forward(self, decoder_x, skip):
        decoder_x = self.up(decoder_x)
        decoder_x = torch.cat([decoder_x, skip], axis=1)
        decoder_x = self.conv(decoder_x)
        return decoder_x


class BuildUnet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = EncoderBlock(3, 64)
        self.e2 = EncoderBlock(64, 128)
        self.e3 = EncoderBlock(128, 256)
        self.e4 = EncoderBlock(256, 512)

        """ Bottleneck """
        self.b = ConvBlock(512, 1024)

        """ Decoder """
        self.d1 = DecoderBlock(1024, 512)
        self.d2 = DecoderBlock(512, 256)
        self.d3 = DecoderBlock(256, 128)
        self.d4 = DecoderBlock(128, 64)

        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, inputs):
        skip_con1, pool1 = self.e1(inputs)
        skip_con2, pool2 = self.e2(pool1)
        skip_con3, pool3 = self.e3(pool2)
        skip_con4, p4 = self.e4(pool3)

        b = self.b(p4)

        d1 = self.d1(b, skip_con4)
        d2 = self.d2(d1, skip_con3)
        d3 = self.d3(d2, skip_con2)
        d4 = self.d4(d3, skip_con1)

        outputs = self.outputs(d4)

        return outputs


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256)) # Batchsize Channel Height With
    f = BuildUnet()
    y = f(x)
    print(y.shape)
