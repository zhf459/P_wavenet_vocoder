# coding: utf-8
from __future__ import with_statement, print_function, absolute_import

import math

import librosa
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from deepvoice3_pytorch.modules import Embedding

from train import build_model
from wavenet_vocoder import receptive_field_size
from wavenet_vocoder.wavenet import _expand_global_features, WaveNet
from .modules import Conv1d1x1, ResidualConv1dGLU, ConvTranspose2d
from .mixture import sample_from_discretized_mix_logistic


class StudentWaveNet(nn.Module):
    """The WaveNet model that supports local and global conditioning.

    Args:
        out_channels (int): Output channels. If input_type is mu-law quantized
          one-hot vecror. this must equal to the quantize channels. Other wise
          num_mixtures x 3 (pi, mu, log_scale).
        layers (int): Number of total layers
        stacks (int): Number of dilation cycles
        residual_channels (int): Residual input / output channels
        gate_channels (int): Gated activation channels.
        skip_out_channels (int): Skip connection channels.
        kernel_size (int): Kernel size of convolution layers.
        dropout (float): Dropout probability.
        cin_channels (int): Local conditioning channels. If negative value is
          set, local conditioning is disabled.
        gin_channels (int): Global conditioning channels. If negative value is
          set, global conditioning is disabled.
        n_speakers (int): Number of speakers. Used only if global conditioning
          is enabled.
        weight_normalization (bool): If True, DeepVoice3-style weight
          normalization is applied.
        upsample_conditional_features (bool): Whether upsampling local
          conditioning features by transposed convolution layers or not.
        upsample_scales (list): List of upsample scale.
          ``np.prod(upsample_scales)`` must equal to hop size. Used only if
          upsample_conditional_features is enabled.
        freq_axis_kernel_size (int): Freq-axis kernel_size for transposed
          convolution layers for upsampling. If you only care about time-axis
          upsampling, set this to 1.
        scalar_input (Bool): If True, scalar input ([-1, 1]) is expected, otherwise
          quantized one-hot vector is expected.
    """

    def __init__(self, out_channels=2,
                 layers=30, stacks=3,
                 iaf_layer_size=[10, 10, 10, 30],
                 # iaf_layer_size=[10, 30],
                 residual_channels=64,
                 gate_channels=64,
                 # skip_out_channels=-1,
                 kernel_size=3, dropout=1 - 0.95,
                 cin_channels=-1, gin_channels=-1, n_speakers=None,
                 weight_normalization=True,
                 upsample_conditional_features=False,
                 upsample_scales=None,
                 freq_axis_kernel_size=3,
                 scalar_input=True,
                 is_student=True,
                 ):
        super(StudentWaveNet, self).__init__()
        self.scalar_input = scalar_input
        self.out_channels = out_channels
        self.cin_channels = cin_channels
        self.is_student = is_student
        self.last_layers = []
        # 噪声
        assert layers % stacks == 0
        layers_per_stack = layers // stacks
        if scalar_input:
            self.first_conv = nn.ModuleList([Conv1d1x1(1, residual_channels) for _ in range(len(iaf_layer_size))])
        else:
            self.first_conv = nn.ModuleList(
                [Conv1d1x1(out_channels, residual_channels) for _ in range(len(iaf_layer_size))])
        self.iaf_layers = nn.ModuleList()  # iaf层
        self.last_layers = nn.ModuleList()

        for layer_size in iaf_layer_size:  # build iaf layers -->4 layers by size 10,10,10,30
            # IAF LAYERS
            iaf_layer = nn.ModuleList()
            for layer in range(layer_size):
                dilation = 2 ** (layer % layers_per_stack)
                conv = ResidualConv1dGLU(
                    residual_channels, gate_channels,
                    kernel_size=kernel_size,
                    bias=True,  # magenda uses bias, but musyoku doesn't
                    dilation=dilation,
                    dropout=dropout,
                    cin_channels=cin_channels,
                    gin_channels=gin_channels,
                    weight_normalization=weight_normalization)
                iaf_layer.append(conv)
            self.iaf_layers.append(iaf_layer)
            self.last_layers.append(nn.ModuleList([  # iaf的最后一层
                nn.ReLU(inplace=True),
                Conv1d1x1(residual_channels, out_channels, weight_normalization=weight_normalization),
                # nn.ReLU(inplace=True),
                # Conv1d1x1(residual_channels, out_channels, weight_normalization=weight_normalization),
            ]))

        if gin_channels > 0:
            assert n_speakers is not None
            self.embed_speakers = Embedding(n_speakers, gin_channels, padding_idx=None, std=0.1)
        else:
            self.embed_speakers = None

        # Upsample conv net
        if upsample_conditional_features:
            self.upsample_conv = nn.ModuleList()
            for s in upsample_scales:
                freq_axis_padding = (freq_axis_kernel_size - 1) // 2
                convt = ConvTranspose2d(1, 1, (freq_axis_kernel_size, s),
                                        padding=(freq_axis_padding, 0),
                                        dilation=1, stride=(1, s),
                                        weight_normalization=weight_normalization)
                self.upsample_conv.append(convt)
                # assuming we use [0, 1] scaled features
                # this should avoid non-negative upsampling output
                self.upsample_conv.append(nn.ReLU(inplace=True))
        else:
            self.upsample_conv = None

        self.receptive_field = receptive_field_size(layers, stacks, kernel_size)

    def has_speaker_embedding(self):
        return self.embed_speakers is not None

    def local_conditioning_enabled(self):
        return self.cin_channels > 0

    def forward(self, z, c=None, g=None, softmax=False):
        """Forward step

        Args:
            x (Variable): One-hot encoded audio signal, shape (B x C x T)
            c (Variable): Local conditioning features, shape (B x C' x T)
            g (Variable): Global conditioning features, shape (B x C'')
            softmax (bool): Whether applies softmax or not.

        Returns:
            Variable: output, shape B x out_channels x T
        """
        # Expand global conditioning features to all time steps
        B, _, T = z.size()

        if g is not None:
            g = self.embed_speakers(g.view(B, -1))
            assert g.dim() == 3
            # (B x gin_channels, 1)
            g = g.transpose(1, 2)
        g_bct = _expand_global_features(B, T, g, bct=True)

        if c is not None and self.upsample_conv is not None:
            # B x 1 x C x T
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
            assert c.size(-1) == z.size(-1)

        # Feed data to network
        mu_tot = Variable(torch.rand(z.size()).fill_(0)).cuda()
        scale_tot = Variable(torch.rand(z.size()).fill_(1).cuda())
        s = []
        m = []
        index = 0
        for first_con, iaf, last_layer in zip(self.first_conv, self.iaf_layers, self.last_layers):  # iaf layer forward
            new_z = first_con(z)
            for f in iaf:
                new_z, h = f(new_z, c, g_bct)
            for f in last_layer:
                new_z = f(new_z)
            mu_f, scale_f = new_z[:, :1, :], torch.exp(new_z[:, 1:, :])
            s.append(scale_f)
            m.append(mu_f)
            z = z * scale_f + mu_f
        for i in range(len(s)):  # update mu_tot and scale_tot base on the paper
            ss = Variable(torch.rand(z.size()).fill_(1).cuda())
            for j in range(i + 1, len(s)):
                ss = ss * s[j]
            mu_tot = mu_tot + m[i] * ss
            scale_tot = scale_tot * s[i]
        return mu_tot, scale_tot

    def clear_buffer(self):
        self.first_conv.clear_buffer()
        for f in self.conv_layers:
            f.clear_buffer()
        for f in self.last_conv_layers:
            try:
                f.clear_buffer()
            except AttributeError:
                pass

    def make_generation_fast_(self):
        def remove_weight_norm(m):
            try:
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(remove_weight_norm)
