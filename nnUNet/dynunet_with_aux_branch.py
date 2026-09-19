from collections.abc import Sequence

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from monai.networks.blocks.dynunet_block import UnetBasicBlock, UnetOutBlock, UnetResBlock, UnetUpBlock
from monai.networks.blocks.dynunet_block import get_conv_layer

__all__ = ["DynUNet", "DynUnet", "Dynunet"]

# up-convolution block for the decoder 
# contains transpose convolution and the UNet Basic Block
class SSUpBlock(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        stride: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
        dropout: tuple | str | float | None = None,
        trans_bias: bool = False,
    ):
        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            dropout=dropout,
            bias=trans_bias,
            act=None,
            norm=None,
            conv_only=False,
            is_transposed=True,
        )
        self.conv_block = UnetBasicBlock(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            dropout=dropout,
            norm_name=norm_name,
            act_name=act_name,
        )

    def forward(self, inp):
        out = self.transp_conv(inp)
        out = self.conv_block(out)
        return out

class DynUNet(nn.Module):

    def __init__(
            self, 
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Sequence[Sequence[int] | int],
            strides: Sequence[Sequence[int] | int],
            upsample_kernel_size: Sequence[Sequence[int] | int],
            filters: Sequence[int] | None = None,
            dropout: tuple | str | float | None = None,
            norm_name: tuple | str = ("INSTANCE", {"affine": True}),
            act_name: tuple | str = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
            deep_supervision: bool = False,
            deep_supr_num: int = 1,
            res_block: bool = False,
            trans_bias: bool = False,):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.strides = strides
        self.upsample_kernel_size = upsample_kernel_size
        self.norm_name = norm_name
        self.act_name = act_name
        self.dropout = dropout
        # allows use of res blocks 
        if res_block:
            self.conv_block = UnetResBlock
        else:
            self.conv_block = UnetBasicBlock
        self.trans_bias = trans_bias
        if filters is not None:
            self.filters = filters
            # check strides are the same length as filters
            self.check_filters()
        else:
            self.filters = [min(2 ** (5 + i), 320 if spatial_dims == 3 else 512) for i in range(len(strides))]
        self.input_block = self.get_input_block()
        self.downsamples = self.get_downsamples()
        self.bottleneck = self.get_bottleneck()
        self.upsamples = self.get_upsamples()
        self.output_block = self.get_output_block(0)
        self.ss_upsamples = self.get_ss_upsamples()
        self.ss_output = self.get_ss_output()

        self.apply(self.initialize_weights)

        self.deep_supervision = deep_supervision
        self.deep_supr_num = deep_supr_num
        if (self.deep_supervision):
            self.deep_supervision_heads = self.get_deep_supervision_heads()

    def forward(self, x, get_reconstruction=False):
        skips = [] # stores all values for skip connections

        # input block 
        x = self.input_block(x)
        skips.append(x)

        # encoder
        for downsample_block in self.downsamples:
            x = downsample_block(x)
            skips.append(x)

        # bottleneck 
        x_bottleneck = self.bottleneck(x)

        # Main task path
        main_num_layers = len(self.upsamples) # number of layers for main task branch 
        seg = x_bottleneck 
        deep_supervision_outputs = []

        for m in range(main_num_layers):
            main_upsample_index = main_num_layers - m - 1
            main_upsample_block = self.upsamples[m] # get upsample block
            skip = skips[main_upsample_index] # get skip connection
            seg = main_upsample_block(seg, skip) # do upsampling with skip connection
            # handle deep supervision
            if (self.deep_supervision and m < self.deep_supr_num):
                # note that each of the deep supervision heads are output blocks
                deep_supervision_outputs.append(
                    self.deep_supervision_heads[m](seg)
                )

        # main task output 
        seg = self.output_block(seg)

        if (self.training and self.deep_supervision):
            outputs = []
            outputs.append(seg)
            for out in deep_supervision_outputs:
                outputs.append(interpolate(out, seg.shape[2:]))
            seg = torch.stack(outputs, dim=1)

        if (not get_reconstruction):
            return seg

        # Self supervision path 
        ss_num_layers = len(self.ss_upsamples) # number of layers for self-supervision task branch 
        reconstruction = x_bottleneck
        for s in range(ss_num_layers):
            ss_upsample_block = self.ss_upsamples[s] # get upsample block
            reconstruction = ss_upsample_block(reconstruction) # get reconstruction

        # reconstruction task output
        reconstruction = self.ss_output(reconstruction)

        return seg, reconstruction

    def check_filters(self):
        # check filters work with the stride sizes
        filters = self.filters
        if len(filters) < len(self.strides):
            raise ValueError("length of filters should be no less than the length of strides.")
        elif len(filters) > len(self.strides):
            print("WARNING: length of filters longer than length of strides")
        else:
            self.filters = filters[: len(self.strides)]

    def get_input_block(self):
        return self.conv_block(
                    self.spatial_dims,
                    self.in_channels,
                    self.filters[0],
                    self.kernel_size[0],
                    self.strides[0],
                    self.norm_name,
                    self.act_name,
                    dropout=self.dropout,
                )

    def get_bottleneck(self):
        return self.conv_block(
                    self.spatial_dims,
                    self.filters[-2],
                    self.filters[-1],
                    self.kernel_size[-1],
                    self.strides[-1],
                    self.norm_name,
                    self.act_name,
                    dropout=self.dropout,
                )

    def get_output_block(self, idx:int):
        return UnetOutBlock(
            self.spatial_dims, 
            self.filters[idx], 
            self.out_channels, 
            dropout=self.dropout)

    def get_upsamples(self):
        inp, out = self.filters[1:][::-1], self.filters[:-1][::-1]
        strides, kernel_size = self.strides[1:][::-1], self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp,  # type: ignore
            out,  # type: ignore
            kernel_size,
            strides,
            UnetUpBlock,  # type: ignore
            upsample_kernel_size,
            trans_bias=self.trans_bias,
        )
    
    def get_downsamples(self):
        inp, out = self.filters[:-2], self.filters[1:-1]
        strides, kernel_size = self.strides[1:-1], self.kernel_size[1:-1]
        return self.get_module_list(inp, out, kernel_size, strides, self.conv_block)  # type: ignore
    
    def get_module_list(
        self,
        in_channels: list[int],
        out_channels: list[int],
        kernel_size: Sequence[Sequence[int] | int],
        strides: Sequence[Sequence[int] | int],
        conv_block: nn.Module,
        upsample_kernel_size: Sequence[Sequence[int] | int] | None = None,
        trans_bias: bool = False,
    ):
        layers = []
        if upsample_kernel_size is not None:
            for in_c, out_c, kernel, stride, up_kernel in zip(
                in_channels, out_channels, kernel_size, strides, upsample_kernel_size
            ):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                    "upsample_kernel_size": up_kernel,
                    "trans_bias": trans_bias,
                }
                layer = conv_block(**params)
                layers.append(layer)
        else:
            for in_c, out_c, kernel, stride in zip(in_channels, out_channels, kernel_size, strides):
                params = {
                    "spatial_dims": self.spatial_dims,
                    "in_channels": in_c,
                    "out_channels": out_c,
                    "kernel_size": kernel,
                    "stride": stride,
                    "norm_name": self.norm_name,
                    "act_name": self.act_name,
                    "dropout": self.dropout,
                }
                layer = conv_block(**params)
                layers.append(layer)
        return nn.ModuleList(layers)

    @staticmethod
    def initialize_weights(module):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            module.weight = nn.init.kaiming_normal_(module.weight, a=0.01)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

    def get_ss_upsamples(self):
        inp = self.filters[1:][::-1]
        out = self.filters[:-1][::-1]
        strides = self.strides[1:][::-1]
        kernel_size = self.kernel_size[1:][::-1]
        upsample_kernel_size = self.upsample_kernel_size[::-1]
        return self.get_module_list(
            inp, 
            out,
            kernel_size,
            strides,
            SSUpBlock,
            upsample_kernel_size,
            trans_bias=self.trans_bias
        )

    def get_ss_output(self):
        return UnetOutBlock(
            self.spatial_dims, 
            self.filters[0], 
            self.in_channels, 
            dropout=self.dropout)

    def get_deep_supervision_heads(self):
        return nn.ModuleList(
            [self.get_output_block(len(self.filters)-2-i) for i in range(self.deep_supr_num)]
        )
