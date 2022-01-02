import torch
from torch import nn
import dxeon as dx
from einops import rearrange

class AttentionConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: float,
    ):
        super().__init__()

        self.reduction_channels = int(in_channels // reduction_ratio)

        self.query_conv = nn.Conv2d(in_channels, self.reduction_channels, 1, 1, 0, bias = True)
        self.key_conv = nn.Conv2d(in_channels, self.reduction_channels, 1, 1, 0, bias = True)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias = True)

        self.gamma = nn.Parameter(torch.tensor([0.]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size()[-2:]
        q = rearrange(self.query_conv(x), 'b c h w -> b c (h w)')
        # .view(bc[0], self.reduction_channels, -1)
        k = rearrange(self.key_conv(x), 'b c h w -> b c (h w)')
        # .view(bc[0], self.reduction_channels, -1)
        v = rearrange(self.value_conv(x), 'b c h w -> b c (h w)')
        # .view(*bc, -1)

        correlations = torch.bmm(q.permute(0, 2, 1), k)
        beta = torch.softmax(correlations, dim = 1) # (-1, reduction_channels, h * w)
        attention = self.gamma * torch.bmm(v, beta) # (-1, in_channels, h * w)

        out = (rearrange(attention, 'b c (h w) -> b c h w', h = h, w = w) + x)

        return out

class AttentionConv3dType0(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_temporal: int,
        reduction_ratio: float,
    ):
        super().__init__()

        self.temporal_attention = AttentionConv2d(in_temporal * in_channels, reduction_ratio)

        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = rearrange(x, 'b c t h w -> b (c t) h w')
        x_ = self.temporal_attention(x_)
        out = (self.gamma * rearrange(x_, 'b (c t) h w -> b c t h w', c = self.in_channels)) + x

        return out

class AttentionConv3dType1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_temporal: int,
        reduction_ratio: float,
    ):
        super().__init__()

        self.temporal_attention = AttentionConv2d(in_temporal, reduction_ratio)

        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ = rearrange(x, 'b c t h w -> (b c) t h w')
        x_ = self.temporal_attention(x_)
        out = (self.gamma * rearrange(x_, '(b c) t h w -> b c t h w', c = self.in_channels)) + x

        return out

class AttentionConv3dType2(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels_reduction_ratio: float,
        in_temporal: int,
    ):
        super().__init__()

        self.channel_attention = nn.ModuleList([AttentionConv2d(in_channels, channels_reduction_ratio) for _ in range(in_temporal)])

        self.in_temporal = in_temporal

        self.gamma = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_channel_attention = []
        for t in range(self.in_temporal):
            x_channel_attention.append(self.channel_attention[t](x[:, :, t, :, :]))

        x_channel_attention = torch.stack(x_channel_attention, dim = 2)

        out = (self.gamma * x_channel_attention) + x

        return out

# class AttentionConv3dType2(nn.Module):
#     def __init__(
#         self,
#         in_channels: int,
#         channels_reduction_ratio: float,
#         in_temporal: int,
#         temporal_reduction_ratio: float,
#         combination = 'separate', # sequential
#     ):
#         super().__init__()

#         self.temporal_attention = nn.ModuleList([AttentionConv2d(in_temporal, temporal_reduction_ratio) for _ in range(in_channels)])
#         self.channel_attention = nn.ModuleList([AttentionConv2d(in_channels, channels_reduction_ratio) for _ in range(in_temporal)])

#         self.in_channels = in_channels
#         self.in_temporal = in_temporal
#         self.combination = combination
        
#         if combination == 'separate':
#             self.alpha = nn.Parameter(torch.tensor(0.2))
#             self.beta = nn.Parameter(torch.tensor(0.2))
#         else:
#             self.gamma = nn.Parameter(torch.tensor(0.2))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x_channel_attention = []
#         for t in range(self.in_temporal):
#             x_channel_attention.append(self.channel_attention[t](x[:, :, t, :, :]))

#         x_channel_attention = torch.stack(x_channel_attention, dim = 2)

#         if self.combination == 'separate':
#             x_temporal_attention = []
#             for c in range(self.in_channels):
#                 x_temporal_attention.append(self.temporal_attention[c](x[:, c, :, :, :]))
            
#             x_temporal_attention = torch.stack(x_temporal_attention, dim = 1)

#             out = (self.alpha * x_channel_attention) + (self.beta * x_temporal_attention) + x

#         elif self.combination == 'sequential':
#             x_temporal_attention = []
#             for c in range(self.in_channels):
#                 x_temporal_attention.append(self.temporal_attention[c](x_channel_attention[:, c, :, :, :]))
            
#             out = (self.gamma * torch.stack(x_temporal_attention, dim = 1)) + x

#         return out

# block = AttentionConv3dType3(512, 64, 8, 4).cuda()
# dx.stats.summarize(block, (8, 512, 8, 16, 24))
# dx.utils.benchmark_performance(block, torch.ones(8, 512, 8, 16, 24).cuda(), 10)

# block = AttentionConv3dType0(512, 4, 64).cpu()
# dx.stats.summarize(block, (8, 512, 4, 16, 24))
# dx.utils.benchmark_performance(block, torch.ones(8, 512, 4, 16, 24).cpu(), 10)

# block = AttentionConv2d(512, 64).cuda()
# dx.stats.summarize(block, (8, 512, 16, 24))
# dx.utils.benchmark_performance(block, torch.ones(8, 512, 16, 24).cuda(), 100)

# print(torch.cuda.memory_summary(device = 0, abbreviated = True))

# def calc_ins_mean_std(x, eps=1e-5):
#     assert (len(x.size()) == 4)

#     var = rearrange(x, 'b c w h -> b c (w h)').var(dim = 2) + eps
#     std = rearrange(var.sqrt(), 'b c -> b c 1 1')
#     mean = rearrange(rearrange(x, 'b c w h -> b c (w h)').mean(dim = 2), 'b c -> b c 1 1')

#     return mean, std

# class SelfNorm(nn.Module):
#     def __init__(self, in_channels, is_two = True):
#         super(SelfNorm, self).__init__()

#         self.g_fc = nn.Conv1d(in_channels, in_channels, kernel_size = 2, bias = False, groups = in_channels)
#         self.g_bn = nn.BatchNorm1d(in_channels)

#         if is_two is True:
#             self.f_fc = nn.Conv1d(in_channels, in_channels, kernel_size = 2, bias = False, groups = in_channels)
#             self.f_bn = nn.BatchNorm1d(in_channels)
#         else:
#             self.f_fc = None

#     def forward(self, x):
#         b, c, _, _ = x.size()

#         mean, std = calc_ins_mean_std(x, eps = 1e-12)

#         statistics = torch.cat((mean.squeeze(3), std.squeeze(3)), -1)
#         print('statistics.shape:', statistics.shape)

#         g_y = self.g_fc(statistics)
#         print('g_y.shape:', g_y.shape)
#         g_y = self.g_bn(g_y)
#         g_y = torch.sigmoid(g_y)
#         g_y = g_y.view(b, c, 1, 1)

#         if self.f_fc is not None:
#             f_y = self.f_fc(statistics)
#             f_y = self.f_bn(f_y)
#             f_y = torch.sigmoid(f_y)
#             f_y = f_y.view(b, c, 1, 1)

#             return x * g_y + mean * (f_y - g_y)
#         else:
#             return x * g_y

# block = SelfNorm(16)
# dx.stats.summarize(block, (8, 16, 64, 64))
# dx.utils.benchmark_performance(block, torch.ones((8, 16, 64, 64)))

class SqueezeAndExcitationBlock3dType0(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_temporal: int,
        reduction_ratio: float = 2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv2d_squeeze_and_excitation = dx.modules.ChannelSqueezeAndExcitationBlock2d(in_temporal, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c t w h -> (b c) t w h')
        x = self.conv2d_squeeze_and_excitation(x)
        out = rearrange(x, '(b c) t w h -> b c t w h', c = self.in_channels)

        return out

class SqueezeAndExcitationBlock3dType1(nn.Module):
    def __init__(
        self,
        in_channels: int,
        in_temporal: int,
        reduction_ratio: float = 2.0,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.conv2d_squeeze_and_excitation = dx.modules.ChannelSqueezeAndExcitationBlock2d(in_temporal * in_channels, reduction_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, 'b c t w h -> b (c t) w h')
        x = self.conv2d_squeeze_and_excitation(x)
        out = rearrange(x, 'b (c t) w h -> b c t w h', c = self.in_channels)

        return out

class SqueezeAndExcitationBlock3dType2(nn.Module):
    def __init__(
        self,
        in_channels,
        reduction_ratio,
        expand_1x1_channels,
        expand_3x3_channels,
        residual = True
    ):
        super().__init__()

        self.residual = residual
        self.in_channels = in_channels
        
        self.relu = nn.ReLU(inplace = True)
        
        self.squeeze = nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size = 1)
        self.squeeze_bn = nn.BatchNorm3d(in_channels // reduction_ratio)
        
        self.expand_1x1 = nn.Conv3d(in_channels // reduction_ratio, expand_1x1_channels, kernel_size = 1)
        self.expand_1x1_bn = nn.BatchNorm3d(expand_1x1_channels)
        
        self.expand_3x3 = nn.Conv3d(in_channels // reduction_ratio, expand_3x3_channels, kernel_size = 3, padding = 1)
        self.expand_3x3_bn = nn.BatchNorm3d(expand_3x3_channels)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand_1x1(out)
        out1 = self.expand_1x1_bn(out1)
        
        out2 = self.expand_3x3(out)
        out2 = self.expand_3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        
        if self.residual:
            out += x

        out = self.relu(out)

        return out

# block = SqueezeAndExcitationBlock3dType2(512, 4)
# dx.stats.summarize(block, (8, 512, 4, 64, 64))
# dx.utils.benchmark_performance(block, torch.ones((8, 512, 4, 64, 64)))

class ConvBlock3D(nn.Module):
	def __init__(
        self,
        in_channels,
        out_channels,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        spatial_downsample: bool = False,
        temporal_downsample: bool = False,
		residual = True,
    ):
		super().__init__()

		self.residual = residual

		_spatial_padding = 1 if spatial_downsample else spatial_kernel_size // 2
		_spatial_stride = 2 if spatial_downsample else 1

		_temporal_padding = 1 if temporal_downsample else temporal_kernel_size // 2
		_temporal_stride = 2 if temporal_downsample else 1

		self.downsampled = spatial_downsample or temporal_downsample

		if not self.downsampled and self.residual:
			assert out_channels == in_channels, f'\n\nin_channels and out_channels must be same for residual connection.\n'

		self.conv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size = (temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
            stride = (_temporal_stride, _spatial_stride, _spatial_stride),
            padding = (_temporal_padding, _spatial_padding, _spatial_padding),
            bias = None
        )
		self.bn = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(True)

	def forward(self, inputs):
		x = self.conv3d(inputs)
		x = self.bn(x)

		if not self.downsampled and self.residual:
			x += inputs

		x = self.relu(x)

		return x

class SeparableConvBlock3D(nn.Module):
	def __init__(
        self,
        in_channels,
        out_channels,
        in_temporal,
        spatial_kernel_size: int = 3,
        temporal_kernel_size: int = 3,
        spatial_downsample: bool = False,
        temporal_downsample: bool = False,
		residual = True,
    ):
		super().__init__()

		self.residual = residual

		_spatial_padding = 1 if spatial_downsample else spatial_kernel_size // 2
		_spatial_stride = 2 if spatial_downsample else 1

		_temporal_padding = 1 if temporal_downsample else temporal_kernel_size // 2
		_temporal_stride = 2 if temporal_downsample else 1

		self.downsampled = spatial_downsample or temporal_downsample

		if not self.downsampled and self.residual:
			assert out_channels == in_channels, f'\n\nin_channels and out_channels must be same for residual connection.\n'

		self.dwconv3d = nn.Conv3d(
            in_channels,
            in_channels,
            kernel_size = (temporal_kernel_size, spatial_kernel_size, spatial_kernel_size),
            stride = (_temporal_stride, _spatial_stride, _spatial_stride),
            padding = (_temporal_padding, _spatial_padding, _spatial_padding),
            bias = None,
			groups = in_channels,
        )
		self.pwconv3d = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size = (in_temporal, 1, 1),
            stride = (1, 1, 1),
            padding = (0, 0, 0),
            bias = None,
			groups = 1,
        )
		self.bn = nn.BatchNorm3d(out_channels)
		self.relu = nn.ReLU(True)

	def forward(self, inputs):
		x = self.dwconv3d(inputs)
		x = self.pwconv3d(x)
		x = self.bn(x)
		
		if (not self.downsampled) and self.residual:
			x += inputs

		x = self.relu(x)

		return x

class R2Plus1D(nn.Module):
	def __init__(
		self,
		in_channels,
		mid_channels,
		out_channels,
		kernel_size: int = 3,
		spatial_downsample: bool = False,
		temporal_downsample: bool = False,
		residual = True,
		final_relu = True,
	):
		super().__init__()

		self.residual = residual

		_spatial_padding = 1 if spatial_downsample else kernel_size // 2
		_spatial_stride = 2 if spatial_downsample else 1

		_temporal_padding = 1 if temporal_downsample else kernel_size // 2
		_temporal_stride = 2 if temporal_downsample else 1

		self.downsampled = spatial_downsample or temporal_downsample

		if not self.downsampled and self.residual:
			assert out_channels == in_channels, f'\n\nin_channels and out_channels must be same for residual connection.\n'

		self.conv_r2 = nn.Conv3d(
			in_channels,
			mid_channels,
			kernel_size = (1, kernel_size, kernel_size),
			stride = (1, _spatial_stride, _spatial_stride),
			padding = (0, _spatial_padding, _spatial_padding),
			bias = None
		)
		self.bn1 = nn.BatchNorm3d(mid_channels)
		self.relu1 = nn.ReLU(True)
		self.conv_r1 = nn.Conv3d(
			mid_channels,
			out_channels,
			kernel_size = (kernel_size, 1, 1),
			stride = (_temporal_stride, 1, 1),
			padding = (_temporal_padding, 0, 0),
			bias = None
		)
		self.bn2 = nn.BatchNorm3d(out_channels)
		self.relu2 = nn.ReLU(True)
		self.final_relu = final_relu

	def forward(self, inputs):
		x = self.conv_r2(inputs)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.conv_r1(x)
		x = self.bn2(x)

		if self.residual:
			if not self.downsampled:
				x += inputs
			else:
				x += dx.F.avg_pool3d(input = inputs, kernel_size = (3, 3, 3), stride = (2, 2, 2), padding = (1, 1, 1))

		if self.final_relu:
			x = self.relu2(x)

		return x

class SeparableR2Plus1D(nn.Module):
	def __init__(
		self,
		in_channels,
		mid_channels,
		out_channels,
		kernel_size: int = 3,
		spatial_downsample: bool = False,
		temporal_downsample: bool = False,
		residual = True
	):
		super().__init__()

		self.residual = residual

		_spatial_padding = 1 if spatial_downsample else kernel_size // 2
		_spatial_stride = 2 if spatial_downsample else 1

		_temporal_padding = 1 if temporal_downsample else kernel_size // 2
		_temporal_stride = 2 if temporal_downsample else 1

		self.downsampled = spatial_downsample or temporal_downsample

		if not self.downsampled and self.residual:
			assert out_channels == in_channels, f'\n\nin_channels and out_channels must be same for residual connection.\n'

		self.dwconv_r2 = nn.Conv3d(
			in_channels,
			in_channels,
			kernel_size = (1, kernel_size, kernel_size),
			stride = (1, _spatial_stride, _spatial_stride),
			padding = (0, _spatial_padding, _spatial_padding),
			bias = None,
			groups = in_channels
		)
		self.pwconv_r2 = nn.Conv3d(
			in_channels,
			mid_channels,
			kernel_size = (1, 1, 1),
			stride = (1, 1, 1),
			padding = (0, 0, 0),
			bias = None,
			groups = 1
		)
		self.bn1 = nn.BatchNorm3d(mid_channels)
		self.relu1 = nn.ReLU(True)
		self.dwconv_r1 = nn.Conv3d(
			mid_channels,
			mid_channels,
			kernel_size = (kernel_size, 1, 1),
			stride = (_temporal_stride, 1, 1),
			padding = (_temporal_padding, 0, 0),
			bias = None,
			groups = mid_channels
		)
		self.pwconv_r1 = nn.Conv3d(
			mid_channels,
			out_channels,
			kernel_size = (1, 1, 1),
			stride = (1, 1, 1),
			padding = (0, 0, 0),
			bias = None,
			groups = 1
		)

		self.bn2 = nn.BatchNorm3d(out_channels)
		self.relu2 = nn.ReLU(True)

	def forward(self, inputs):
		x = self.dwconv_r2(inputs)
		x = self.pwconv_r2(x)
		x = self.bn1(x)
		x = self.relu1(x)
		x = self.dwconv_r1(x)
		x = self.pwconv_r1(x)
		x = self.bn2(x)

		if self.residual:
			if not self.downsampled:
				x += inputs
			else:
				x += dx.F.avg_pool3d(input = inputs, kernel_size = (2, 2, 2), stride = (2, 2, 2), padding = 0)

		x = self.relu2(x)

		return x
    
class PretrainedModule(nn.Module):
    def __init__(self):
        super().__init__()

        # torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

        self.blocks = torch.hub.load("facebookresearch/pytorchvideo:main", "x3d_l", pretrained = True).blocks[:-2]

        # self.pretrained = models.video.mc3_18(pretrained = False, progress = True)

        # print(_pretrained.stem)
        # print(_pretrained.layer1)
        # print(_pretrained.layer2)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        # x = self.pretrained.stem(x)
        # x = self.pretrained.layer1(x)
        # x = self.pretrained.layer2(x)

        return x
