import torch
import torch.nn as nn
import functools
from .blocks import *

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, nactors, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.nactors = nactors
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.inc = Inconv(input_nc, ngf, norm_layer, use_bias)
        self.down1 = Down(ngf, ngf * 2, norm_layer, use_bias)
        self.down2 = Down(ngf * 2, ngf * 2, norm_layer, use_bias)
        self.encode = nn.Sequential(
            self.inc,
            self.down1,
            self.down2
        )

        model = []
        bottle_dim = ngf * 2 * nactors
        for i in range(n_blocks):
            # modify here for cat 4 => 8
            model += [ResBlock(bottle_dim, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        self.resblocks = nn.Sequential(*model)
        self.up1 = Up(bottle_dim, ngf * 2, norm_layer, use_bias)
        self.up2 = Up(ngf * 2, ngf, norm_layer, use_bias)

        self.outc = Outconv(ngf, output_nc)

    def forward(self, inputs):
        out = {}
        latent = []
        for i in range(self.nactors):
            x = inputs[:, i, ...]
            l = self.encode(x)
            latent.append(l)
        out['d2'] = torch.cat(latent, dim=1)
        # update the input channels
        out['bottle'] = self.resblocks(out['d2'])
        out['u1'] = self.up1(out['bottle'])
        out['u2'] = self.up2(out['u1'])

        return self.outc(out['u2'])
