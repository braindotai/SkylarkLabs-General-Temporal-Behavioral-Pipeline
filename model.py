import torch
from torch import nn
from torch.nn.modules.flatten import Flatten
from torchvision import models
from pytorch_lightning import LightningModule
import dxeon as dx
import modules

class AvgPool3d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, c, t, h, w = x.size()
        x = x.sum(-1).sum(-1).sum(-1) / (t * h * w)
        return x

class ConvSEAttentionBlock(nn.Module):
    def __init__(self, conv_block, conv_args, attention_block, attention_args, se_block, se_args, prob = None):
        super().__init__()

        self.conv = conv_block(**conv_args)
        self.attention = attention_block(**attention_args) if attention_args else None
        self.se = se_block(**se_args) if se_args else None
        self.downsample = conv_args['spatial_downsample']
        self.different_channels = conv_args['in_channels'] != conv_args['out_channels']
        self.rv = torch.distributions.Bernoulli(probs = torch.Tensor([0.5]))
        self.prob = prob
    
    def forward(self, inputs):
        if self.training:
            print('training...')

            if self.rv.sample() == 1 or self.downsample or self.different_channels:
                for param in self.parameters():
                    param.requires_grad = True
                
                out = self._forward(inputs)
            else:
                for param in self.parameters():
                    param.requires_grad = False
                print('as is')
                out = inputs
        else:
            out = self._forward(inputs)
            
        return out
    
    def _forward(self, x):
        x = self.conv(x)

        if self.attention:
            x = self.attention(x)
        
        if self.se:
            x = self.se(x)

        return x
        

class Model(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        # [(4, 128, False)]
        if self.hparams.separable:
            self._conv_block = modules.SeparableR2Plus1D
        else:
            self._conv_block = modules.R2Plus1D

        self._attention_block = getattr(modules, f'AttentionConv3dType{self.hparams.attention_type}')
        self._se_block = getattr(modules, f'SqueezeAndExcitationBlock3dType{self.hparams.se_type}')

        self.pretrained_features = modules.PretrainedModule() # -> (16, 96, 16, 12, 16)

        in_channels = 96
        out_channels = 96
        in_temporal = 16

        self.blocks = nn.Sequential()
        self.auxillary_blocks = nn.ModuleList()

        for block_idx, block_config in enumerate(self.hparams.blocks_config):
            out_channels = block_config[1]
            has_se = block_config[3]
            has_attention = block_config[4]

            block = nn.Sequential()

            for layer_idx in range(block_config[0]):
                downsample = block_config[2] and layer_idx == 0
                
                conv_args = {
                    'in_channels': in_channels,
                    'mid_channels': in_channels,
                    'out_channels': out_channels,
                    'spatial_downsample': downsample,
                    'temporal_downsample': downsample,
                    'residual': out_channels == in_channels
                }
                
                if downsample:
                    in_temporal = int(in_temporal / 2)
                
                if has_attention:
                    if self.hparams.attention_type in (0, 1):
                        attention_args = {
                            'in_channels': out_channels,
                            'in_temporal': in_temporal,
                            'reduction_ratio': 4,
                        }

                    elif self.hparams.attention_type == 2:
                        attention_args = {
                            'in_channels': out_channels,
                            'channels_reduction_ratio': 4,
                            'in_temporal': in_temporal,
                        }
                else:
                    attention_args = None

                if has_se:
                    if self.hparams.se_type in (0, 1):
                        se_args = {
                            'in_channels': out_channels,
                            'in_temporal': in_temporal,
                            'reduction_ratio': 2.5,
                        }
                    elif self.hparams.se_type == 2:
                        se_args = {
                            'in_channels': out_channels,
                            'reduction_ratio': 4,
                            'expand_1x1_channels': int(out_channels // 2),
                            'expand_3x3_channels': int(out_channels // 2),
                            'residual': True,
                        }
                else:
                    se_args = None
                
                in_channels = out_channels

                block.add_module(
                    name = f'block_{block_idx}_{layer_idx}',
                    module = self.make_block(
                        conv_args = conv_args,
                        attention_args = attention_args,
                        se_args = se_args,
                    )
                )
            
            self.blocks.add_module(name = f'block_{block_idx}', module = block)

            if (block_idx + 1) % self.hparams.auxillary_idx == 0:
                self.auxillary_blocks.append(self.auxillary_block(out_channels, in_temporal))
        
    def forward(self, x):
        x = self.pretrained_features(x)
        aux_idx = 0
        aux_features = []
        
        for block_idx, block in enumerate(self.blocks):
            x = block(x)

            if (block_idx + 1) % self.hparams.auxillary_idx == 0:
                aux_features.append(self.auxillary_blocks[aux_idx](x))
                aux_idx += 1
        
        return aux_features

    def make_block(
        self,
        conv_args,
        attention_args,
        se_args,
    ) -> nn.Module:

        # block = nn.Sequential()
        # block.add_module(f'conv', self._conv_block(**conv_args))
        
        # if attention_args is not None:
        #     block.add_module(f'attention', self._attention_block(**attention_args))
        
        # if se_args is not None:
        #     block.add_module(f'se', self._se_block(**se_args))

        return ConvSEAttentionBlock(
            self._conv_block, conv_args,
            self._attention_block, attention_args,
            self._se_block, se_args,
        )
    
    def auxillary_block(
        self,
        in_channels: int,
        in_temporal: int,
    ) -> nn.Module:

        return nn.Sequential(
            self._conv_block(in_channels, in_channels, in_channels),
            self._se_block(in_channels, in_temporal) if self.hparams.se_type in (0, 1) else self._se_block(in_channels, 2, in_channels // 2, in_channels // 2),
            self._conv_block(in_channels, in_channels, self.hparams.num_classses, residual = False),
            # lambda x: x.sum(-1).sum(-1).sum(-1), # -1, c, t, w, h
            AvgPool3d(),
            nn.Flatten()
        )

model = Model(
    blocks_config = [
        (2, 96, False, True, False),
        (4, 128, False, True, True),
        (3, 256, True, True, True),
        (2, 256, True, True, True)],
    separable = True,
    se_type = 2, # 0 < 2 < 1
    attention_type = 1, # 1 < 2 < 0
    auxillary_idx = 1,
    num_classses = 600,
).cuda()
model.train()

size = (16, 3, 16, 192, 256)
# with torch.no_grad():
#     print(model(torch.ones(*size).cuda())[0].shape)

dx.stats.summarize(model, size)
# dx.utils.benchmark_performance(model, torch.ones(*size).cuda())
# dx.utils.GenerateONNXModel(model, onnx_path = 'model.onnx', input_shape = size)
# dx.viz.model(model, size)