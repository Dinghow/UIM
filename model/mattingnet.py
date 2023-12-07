import torch
import torch.nn as nn
from   torch.nn import Parameter
from   torch.autograd import Variable
import torch.nn.functional as F
import logging
import numpy as np
from util.util import cross_entropy_loss


class conv2DBatchNormRelu(nn.Module):
    def __init__(
            self,
            in_channels,
            n_filters,
            k_size,
            stride,
            padding,
            bias=True,
            dilation=1,
            with_bn=True,
            with_relu=True,
            BatchNorm=nn.BatchNorm2d,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels),
                             int(n_filters),
                             kernel_size=k_size,
                             padding=padding,
                             stride=stride,
                             bias=bias,
                             dilation=dilation, )

        if with_bn:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, BatchNorm(int(n_filters)), nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod, BatchNorm(int(n_filters)))
        else:
            if with_relu:
                self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))
            else:
                self.cbr_unit = nn.Sequential(conv_mod)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class ResBlock(nn.Module):
    def __init__(self, in_size, out_size, stride, BatchNorm, with_bn, expansion=4, shortcut=None):
        super(ResBlock, self).__init__()
        self.shortcut = shortcut
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, k_size=1, stride=1, padding=0, BatchNorm=BatchNorm, with_bn=with_bn, bias=False)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, k_size=3, stride=stride, padding=1, BatchNorm=BatchNorm, with_bn=with_bn, bias=False)
        self.conv3 = conv2DBatchNormRelu(out_size, out_size*expansion, k_size=1, stride=1, padding=0, BatchNorm=BatchNorm, with_bn=with_bn, with_relu=False, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        residual = inputs if self.shortcut is None else self.shortcut(inputs)
        outputs += residual
        return self.relu(outputs)


def make_layer(in_channel, out_channel, stride, BatchNorm, with_bn, expansion=4, block_num=3):
    shortcut = None
    if stride != 1 or in_channel != out_channel*expansion:
        shortcut = conv2DBatchNormRelu(in_channel, out_channel*expansion, k_size=1, stride=stride, padding=0, BatchNorm=BatchNorm, with_bn=with_bn, with_relu=False, bias=False)
    layers = list()
    layers.append(ResBlock(in_channel, out_channel, stride=stride, BatchNorm=BatchNorm, with_bn=with_bn, expansion=expansion, shortcut=shortcut))
    for i in range(1, block_num):
        layers.append(ResBlock(out_channel*expansion, out_channel, stride=1, BatchNorm=BatchNorm, with_bn=with_bn, expansion=expansion, shortcut=None))
    return nn.Sequential(*layers)


class ResNetEncoder(nn.Module):
    def __init__(self, n_classes=1, in_channels=4, pretrain=True,
                 BatchNorm=nn.BatchNorm2d, with_bn=True):
        super(ResNetEncoder, self).__init__()

        self.in_channels = in_channels
        self.pretrain = pretrain

        self.pre = conv2DBatchNormRelu(in_channels, 64, k_size=7, stride=1, padding=3, BatchNorm=BatchNorm, with_bn=with_bn, bias=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, return_indices=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, return_indices=True)
        self.layer1 = make_layer(64  , 64 , stride=1, BatchNorm=BatchNorm, with_bn=with_bn, expansion=4, block_num=3)
        self.layer2 = make_layer(256 , 128, stride=1, BatchNorm=BatchNorm, with_bn=with_bn, expansion=4, block_num=4)
        self.layer3 = make_layer(512 , 256, stride=1, BatchNorm=BatchNorm, with_bn=with_bn, expansion=4, block_num=6)
        self.layer4 = make_layer(1024, 512, stride=1, BatchNorm=BatchNorm, with_bn=with_bn, expansion=4, block_num=3)

    def forward(self, inputs):
        # inputs: [N, 4, 320, 320]
        down1 = self.pre(inputs)
        unpool_shape1 = down1.size()
        down1, indices_1 = self.maxpool1(down1)

        down2 = self.layer1(down1)
        unpool_shape2 = down2.size()
        down2, indices_2 = self.maxpool2(down2)

        down3 = self.layer2(down2)
        unpool_shape3 = down3.size()
        down3, indices_3 = self.maxpool2(down3)

        down4 = self.layer3(down3)
        unpool_shape4 = down4.size()
        down4, indices_4 = self.maxpool2(down4)

        down5 = self.layer4(down4)
        unpool_shape5 = down5.size()
        down5, indices_5 = self.maxpool2(down5)
        return down1, down2, down3, down4, down5, indices_1, indices_2, indices_3, indices_4, indices_5, unpool_shape1, unpool_shape2, unpool_shape3, unpool_shape4, unpool_shape5

    def init_resnet50_params(self, resnet50):
        blocks = [self.pre, self.layer1, self.layer2, self.layer3, self.layer4]
        resblocks = [resnet50.conv1, resnet50.layer1, resnet50.layer2, resnet50.layer3, resnet50.layer4]

        resnet50_layers = []
        merged_layers = []
        for idx, conv_block in enumerate(resblocks):
            if idx < 1:
                if isinstance(conv_block, nn.Conv2d):
                    resnet50_layers.append(conv_block)
            else:
                for idy, bottleblock in enumerate(conv_block):
                    if idy == 0:
                        units = [bottleblock.downsample[0], bottleblock.conv1, bottleblock.conv2, bottleblock.conv3]
                    else:
                        units = [bottleblock.conv1, bottleblock.conv2, bottleblock.conv3]
                    for _layer in units:
                        if isinstance(_layer, nn.Conv2d):
                            resnet50_layers.append(_layer)

        for idx, conv_block in enumerate(blocks):
            if idx < 1:
                units = [conv_block.cbr_unit]
                for unit in units:
                    for _layer in unit:
                        if isinstance(_layer, nn.Conv2d):
                            merged_layers.append(_layer)
            else:
                for idy, bottleblock in enumerate(conv_block):
                    if idy == 0:
                        units = [bottleblock.shortcut.cbr_unit, bottleblock.conv1.cbr_unit, bottleblock.conv2.cbr_unit, bottleblock.conv3.cbr_unit]
                    else:
                        units = [bottleblock.conv1.cbr_unit, bottleblock.conv2.cbr_unit, bottleblock.conv3.cbr_unit]
                    for unit in units:
                        for _layer in unit:
                            if isinstance(_layer, nn.Conv2d):
                                merged_layers.append(_layer)

        assert (len(resnet50_layers) == len(merged_layers))
        for l1, l2 in zip(resnet50_layers, merged_layers):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                if l1.weight.size() == l2.weight.size():
                    l2.weight.data = l1.weight.data
                else:
                    # conv1 from 64*3*7*7 to 64*in_channel*7*7
                    extend_channel = self.in_channels - 3
                    l2.weight.data = torch.cat((l1.weight.data, torch.zeros(64, extend_channel, 7, 7)), 1)


# ======================= Matting ========================= #
def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class MAFModule(nn.Module):
    def __init__(self, features, sizes=(1, 2, 3, 6)):
        super(MAFModule, self).__init__()
        self.size_num = len(sizes) + 1
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.weights = nn.ModuleList([self._make_weight(features, 8) for i in range(self.size_num)])
        self.weight_norm = nn.Conv2d(8*self.size_num, self.size_num, 1, 1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(features)
        relu = nn.LeakyReLU(0.1)
        return nn.Sequential(prior, conv, bn, relu)

    def _make_weight(self, features, weight_channel):
        conv = nn.Conv2d(features, weight_channel, 1, 1)
        bn = nn.BatchNorm2d(weight_channel)
        relu = nn.LeakyReLU(0.1)
        return nn.Sequential(conv, bn, relu)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear',align_corners = True) for stage in self.stages] + [feats]
        level_weights = [self.weights[i](priors[i]) for i in range(self.size_num)]
        level_weight_norm = self.weight_norm(torch.cat(level_weights, 1))
        level_weight_norm = F.softmax(level_weight_norm, dim=1)
        features_norm = [level_weight_norm[:, i:i+1, :,:]*priors[i] for i in range(self.size_num)]
        bottle = features_norm[0]
        for i in range(1, self.size_num):
            bottle += features_norm[i]
        return self.relu(bottle)


class BasicBlockE(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None):
        super(BasicBlockE, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.activation = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet50Encoder(nn.Module):
    """
    Implement and pre-train on ImageNet with the tricks from
    https://arxiv.org/abs/1812.01187
    without the mix-up part.
    """

    def __init__(self, block, layers, norm_layer=None, in_channels=6):
        super(ResNet50Encoder, self).__init__()
        self.logger = logging.getLogger("Logger")
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.in_channels = in_channels
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.activation = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # change the name from layer_bottleneck to layer4
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.xavier_uniform_(m.weight)
        #     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677

        for m in self.modules():
            if isinstance(m, BasicBlockE):
                nn.init.constant_(m.bn2.weight, 0)

        self.logger.debug("encoder conv1 weight shape: {}".format(str(self.conv1.weight.data.shape)))
        self.conv1.weight.data[:,3:,:,:] = 0

        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x1 = self.activation(x) # N x 32 x 256 x 256
        x = self.conv3(x1)
        x = self.bn3(x)
        x2 = self.activation(x) # N x 64 x 128 x 128

        x3 = self.layer1(x2) # N x 64 x 128 x 128
        x4 = self.layer2(x3) # N x 128 x 64 x 64
        x5 = self.layer3(x4) # N x 256 x 32 x 32
        x = self.layer4(x5) # N x 512 x 16 x 16

        return x, (x1, x2, x3, x4, x5)


class ResNet50ShortcutEncoder(ResNet50Encoder):

    def __init__(self, block, layers, norm_layer=None, in_channels=6):
        super(ResNet50ShortcutEncoder, self).__init__(block, layers, norm_layer, in_channels)
        first_inplane = in_channels
        self.shortcut_inplane = [first_inplane, 64, 64, 128, 256]
        self.shortcut_plane = [32, 64, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):
        # x: N x 6 x 512 x 512
        out = self.conv1(x)
        out = self.bn1(out)
        x1 = self.activation(out) # N x 64 x 256 x 256
        out = self.maxpool(x1)
        x2 = self.layer1(out) # N x 64 x 128 x 128
        x3= self.layer2(x2) # N x 128 x 64 x 64
        x4 = self.layer3(x3) # N x 256 x 32 x 32
        out = self.layer4(x4) # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x) # input image and trimap, N x 32 x 512 x 512
        fea2 = self.shortcut[1](x1) # N x 64 x 256 x 256
        fea3 = self.shortcut[2](x2) # N x 64 x 128 x 128
        fea4 = self.shortcut[3](x3) # N x 128 x 64 x 64
        fea5 = self.shortcut[4](x4) # N x 256 x 32 x 32

        return out, (fea1, fea2, fea3, fea4, fea5)

    def init_resnet50_params(self, checkpoint_path):
        """
        Load imagenet pretrained resnet
        Add zeros channel to the first convolution layer
        """
        state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage.cuda())
        # state_dict = remove_prefix_state_dict(checkpoint['state_dict'])
        for key, value in state_dict.items():
            state_dict[key] = state_dict[key].float()

        logger = logging.getLogger("Logger")
        logger.debug("Imagenet pretrained keys({}):".format(len(state_dict.keys())))
        logger.debug("Generator keys({}):".format(len(self.state_dict().keys())))
        logger.debug("Intersection keys({}):".format(
            len(set(self.state_dict().keys()) & set(state_dict.keys()))))
        intersection_keys = set(self.state_dict().keys()) & set(state_dict.keys())
        init_keys = set()
        for i, key in enumerate(intersection_keys):
            if state_dict[key].size() == self.state_dict()[key].size():
                init_keys.add(key)
            else:
                logger.debug("{}({}=>{}) incompatible".format(key, state_dict[key].size(),
                                                              self.state_dict()[key].size()))
        logger.debug("{} keys have been initialized as resnet34".format(len(init_keys)))
        init_state_dict = {key: value for key, value in state_dict.items() if key in init_keys}
        init_state_dict['conv1.weight'] = torch.cat((state_dict['conv1.weight'], torch.zeros(64, self.in_channels - 3, 7, 7).cuda()), 1)
        logger.debug("set the conv1.weight of trimap channel(4-6) to be zeros")

        self.load_state_dict(init_state_dict, strict=False)


class BasicBlockD(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, upsample=None, norm_layer=None):
        super(BasicBlockD, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.stride = stride
        conv = conv3x3
        # change convtranspose2d to upsample+conv
        if self.stride > 1:
            self.conv1=nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=self.stride),
                conv(inplanes, inplanes)
            )
            # self.conv1 = nn.ConvTranspose2d(inplanes, inplanes, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            self.conv1 = conv(inplanes, inplanes)
        self.bn1 = norm_layer(inplanes)
        self.activation = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv(inplanes, planes)
        self.bn2 = norm_layer(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.activation(out)

        return out


class ResNet50Decoder(nn.Module):

    def __init__(self, block, layers, norm_layer=None, n_classes=1):
        self.logger = logging.getLogger("Logger")
        super(ResNet50Decoder, self).__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.kernel_size = 3
        self.inplanes = 512 if layers[0] > 0 else 256

        # self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv1 = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                   nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.bn1 = norm_layer(32)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(32, n_classes, kernel_size=self.kernel_size, stride=1, padding=self.kernel_size//2)

        # self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.layer1 = self._make_layer(block, 256, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        for m in self.modules():
            if isinstance(m, BasicBlockD):
                nn.init.constant_(m.bn2.weight, 0)

        self.logger.debug(self)

    def _make_layer(self, block, planes, blocks, stride=1):
        if blocks == 0:
            return nn.Sequential(nn.Identity())
        norm_layer = self._norm_layer
        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.UpsamplingNearest2d(scale_factor=2),
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )
        elif self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion),
                norm_layer(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, upsample, norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x, mid_fea):
        x = self.layer1(x) # N x 256 x 32 x 32
        x = self.layer2(x) # N x 128 x 64 x 64
        x = self.layer3(x) # N x 64 x 128 x 128
        x = self.layer4(x) # N x 32 x 256 x 256
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)

        alpha = self.sigmoid(x)

        return alpha, None


class ResNet50ShortcutDecoder(ResNet50Decoder):

    def __init__(self, block, layers, norm_layer=None, n_classes=1):
        super(ResNet50ShortcutDecoder, self).__init__(block, layers, norm_layer, n_classes)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea
        x = self.layer1(x) + fea5
        x = self.layer2(x) + fea4
        x = self.layer3(x) + fea3
        x = self.layer4(x) + fea2
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = self.sigmoid(x)

        return alpha


class ResNet50ShortcutDecoder_fusion(ResNet50Decoder):
    """ Two conv layer fusion, with general saliency module """
    def __init__(self, block, layers, norm_layer=None, n_classes=1):
        super(ResNet50ShortcutDecoder_fusion, self).__init__(block, layers, norm_layer, n_classes)
        self.conv1_extend = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
                                nn.Conv2d(64+1, 32, kernel_size=3, stride=1, padding=1, bias=False))
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.gs1 = MAFModule(256, (1, 3, 5))
        self.gs2 = MAFModule(128, (1, 3, 5))
        self.gs3 = MAFModule(64, (1, 3, 5))
        self.gs4 = MAFModule(64, (1, 3, 5))

    def forward(self, x, mid_fea, mask_pred=None):
        fea1, fea2, fea3, fea4, fea5 = mid_fea
        x = self.layer1(x) + fea5
        x = self.gs1(x)
        x = self.layer2(x) + fea4
        x = self.gs2(x)
        x = self.layer3(x) + fea3
        x = self.gs3(x)
        x = self.layer4(x) + fea2
        x = self.gs4(x)
        if mask_pred is not None:
            mask_pred_resize = self.maxpool(mask_pred)
            x = torch.cat((x, mask_pred_resize), dim=1)
            x = self.conv1_extend(x)
        else:
            x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = self.sigmoid(x)

        return alpha


class Unified_Interactive_Matting(nn.Module):
    def __init__(self, n_classes, in_channels, encoder_layers, decoder_layers,
                 BatchNorm=None, mask_criterion=None, matting_criterion=None, fusion_method='light_head'):
        super(Unified_Interactive_Matting, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.fusion_method = fusion_method
        self.encoder = ResNet50ShortcutEncoder(BasicBlockE, encoder_layers, BatchNorm, in_channels)
        self.gs_module = MAFModule(512, (1, 3, 5))
        self.mask_decoder = ResNet50ShortcutDecoder(BasicBlockD, decoder_layers, BatchNorm, n_classes)
        self.matting_decoder = ResNet50ShortcutDecoder_fusion(BasicBlockD, decoder_layers, BatchNorm, n_classes)            
        self.mask_criterion = mask_criterion
        self.matting_criterion = matting_criterion

    def forward(self, image, interactive=None, gt=None, mask=None):
        if interactive is not None:
            inp = torch.cat((image, interactive), dim=1)
        else:
            inp = image
        embedding, mid_fea = self.encoder(inp)
        gs = self.gs_module(embedding)
        mask_pred = self.mask_decoder(gs, mid_fea)
        alpha = self.matting_decoder(gs, mid_fea, mask_pred)
        if self.training:
            mask_loss = self.mask_criterion(mask_pred, mask)
            alpha_loss = self.matting_criterion(alpha, gt, reduction = 'sum') / (np.prod(gt.size()) + 1e-8)
            return alpha, mask_loss, alpha_loss
        else:
            return alpha, mask_pred


class Unified_Interactive_Matting_trimap(nn.Module):
    """ UIM model only focus on unknown region """
    def __init__(self, n_classes, in_channels, encoder_layers, decoder_layers,
                 BatchNorm=None, mask_criterion=None, matting_criterion=None, fusion_method='in_decoder'):
        super(Unified_Interactive_Matting_trimap, self).__init__()
        self.n_classes = n_classes
        self.in_channels = in_channels
        self.fusion_method = fusion_method
        self.encoder = ResNet50ShortcutEncoder(BasicBlockE, encoder_layers, BatchNorm, in_channels)
        self.gs_module = MAFModule(512, (1, 3, 5))
        self.mask_decoder = ResNet50ShortcutDecoder(BasicBlockD, decoder_layers, BatchNorm, n_classes)
        self.matting_decoder = ResNet50ShortcutDecoder_fusion(BasicBlockD, decoder_layers, BatchNorm, n_classes)            
        self.mask_criterion = mask_criterion
        self.matting_criterion = matting_criterion

    def forward(self, image, interactive=None, gt=None, mask=None, trimap=None):
        if interactive is not None:
            inp = torch.cat((image, interactive), dim=1)
        else:
            inp = image
        embedding, mid_fea = self.encoder(inp)
        gs = self.gs_module(embedding)
        mask_pred = self.mask_decoder(gs, mid_fea)
        alpha = self.matting_decoder(gs, mid_fea, mask_pred)
        if self.training:
            weight = trimap[:, 1:2, :, :].float()
            mask_loss = self.mask_criterion(mask_pred, mask)
            alpha_loss = self.matting_criterion(alpha * weight, gt * weight, reduction = 'sum') / (np.prod(gt.size()) + 1e-8)
            return alpha, mask_loss, alpha_loss
        else:
            return alpha