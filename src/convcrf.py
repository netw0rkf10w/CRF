"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import os
import sys

import numpy as np
import math

import logging
import warnings

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)

try:
    import pyinn as P
    has_pyinn = True
except ImportError:
    #  PyInn is required to use our cuda based message-passing implementation
    #  Torch 0.4 provides a im2col operation, which will be used instead.
    #  It is ~15% slower.
    has_pyinn = False
    pass

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter

import torch.nn.functional as F


def _get_ind(dz):
    if dz == 0:
        return 0, 0
    if dz < 0:
        return 0, -dz
    if dz > 0:
        return dz, 0


def _negative(dz):
    """
    Computes -dz for numpy indexing. Goal is to use as in array[i:-dz].

    However, if dz=0 this indexing does not work.
    None needs to be used instead.
    """
    if dz == 0:
        return None
    else:
        return -dz


# def create_position_feats(shape, bs=1):

#     # create mesh
#     hcord_range = [range(s) for s in shape]
#     mesh = np.array(np.meshgrid(*hcord_range, indexing='ij'),
#                     dtype=np.float32)

#     mesh = torch.from_numpy(mesh)

#     if type(mesh) is Parameter:
#         return torch.stack(bs * [mesh])
#     else:
#         return torch.stack(bs * [Variable(mesh)])


def create_position_feats(shape, bs=1):
    # create mesh
    mesh = [torch.arange(s, dtype=torch.float32) for s in shape]
    mesh = torch.meshgrid(*mesh)
    mesh = torch.stack(mesh)
    # print(f'mesh = {mesh.shape}')
    mesh = torch.stack(bs * [mesh])
    # print(f'mesh = {mesh.shape}')
    return mesh


def create_conv_filters(features, filter_size, blur=1):
    """Create Gaussian output from input features (e.g. xy coordinates or RGB)

    Shape:
        features: (N, d, H, W)
    """
    assert filter_size%2 == 1
    span = filter_size//2

    bs, d, h, w = features.shape

    if blur > 1:
        off_0 = (blur - h % blur) % blur
        off_1 = (blur - w % blur) % blur
        pad_0 = math.ceil(off_0 / 2)
        pad_1 = math.ceil(off_1 / 2)
        if blur == 2:
            assert(pad_0 == h % 2)
            assert(pad_1 == w % 2)

        features = torch.nn.functional.avg_pool2d(features,
                                                    kernel_size=blur,
                                                    padding=(pad_0, pad_1),
                                                    count_include_pad=False)
        
        # print(f'features after blur: {features.shape}')
        assert features.shape[2] == math.ceil(h / blur)
        assert features.shape[3] == math.ceil(w / blur)

    # Get the new shape
    bs, d, h, w = features.shape

    gaussian = Variable(features.data.new(bs, filter_size, filter_size, h, w).fill_(0))

    for dx in range(-span, span + 1):
        for dy in range(-span, span + 1):

            dx1, dx2 = _get_ind(dx)
            dy1, dy2 = _get_ind(dy)

            feat_t = features[:, :, dx1:_negative(dx2), dy1:_negative(dy2)]
            feat_t2 = features[:, :, dx2:_negative(dx1), dy2:_negative(dy1)] # NOQA

            diff = feat_t - feat_t2
            diff = diff * diff
            diff = torch.exp(torch.sum(-0.5 * diff, dim=1))

            gaussian[:, dx + span, dy + span,
                        dx2:_negative(dx1), dy2:_negative(dy1)] = diff

    return gaussian.view(bs, 1, filter_size, filter_size, h, w)


def perform_filtering(inputs, gaussian, blur=1, pyinn=False, normalize=True, verbose=False):
    # print(f'inputs: {inputs.shape}')
    filter_size = gaussian.shape[2]
    span = filter_size//2
    input_shape = inputs.shape
    bs, num_channels, h, w = inputs.shape

    if normalize:
        norm = _get_norm(gaussian, h, w, blur=blur)
        inputs = inputs / (norm + 1e-20)

    if blur > 1:
        off_0 = (blur - h % blur) % blur
        off_1 = (blur - w % blur) % blur
        pad_0 = int(math.ceil(off_0 / 2))
        pad_1 = int(math.ceil(off_1 / 2))
        # print(f'pad_0 = {pad_0}, pad_1 = {pad_1}')
        inputs = torch.nn.functional.avg_pool2d(inputs,
                                                kernel_size=blur,
                                                padding=(pad_0, pad_1),
                                                count_include_pad=False)
       
        assert inputs.shape[2] == math.ceil(h / blur)
        assert inputs.shape[3] == math.ceil(w / blur)

    # Get the new shape
    bs, d, h, w = inputs.shape

    # if verbose:
    #     show_memusage(name="Init")

    if pyinn:
        x = P.im2col(inputs, filter_size, 1, span)
    else:
        # An alternative implementation of num2col.
        #
        # This has implementation uses the torch 0.4 im2col operation.
        # This implementation was not avaible when we did the experiments
        # published in our paper. So less "testing" has been done.
        #
        # It is around ~20% slower then the pyinn implementation but
        # easier to use as it removes a dependency.
        x = F.unfold(inputs, filter_size, 1, span)
        x = x.view(bs, num_channels, filter_size, filter_size, h, w)

    k_sqr = filter_size * filter_size

    # if verbose:
    #     show_memusage(name="Im2Col")

    # print(f'inputs device = {inputs.device}')
    # print(f'gaussian device = {gaussian.device}, input_col device = {input_col.device}')
    if verbose:
        print(f'gaussian shape = {gaussian.shape}, x = {x.shape}')
    x = gaussian * x
    # if verbose:
    #     show_memusage(name="Product")

    x = x.view([bs, num_channels, k_sqr, h, w])

    x = x.sum(2)

    # if verbose:
    #     show_memusage(name="FinalNorm")

    if blur > 1:
        # original shape
        in_0 = input_shape[-2]
        in_1 = input_shape[-1]
        x = x.view(bs, num_channels, h, w)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Suppress warning regarding corner alignment
            x = torch.nn.functional.upsample(x,scale_factor=blur, mode='bilinear')
            # print(f'message upscaled = {message.shape}')
        x = x[:, :, pad_0:pad_0 + in_0, pad_1:in_1 + pad_1]
        x = x.contiguous()
        # print(f'message = {message.shape}')
        x = x.view(input_shape)

    if normalize:
        x = x / (norm + 1e-20)

    return x


def _get_norm(gaussian, height, width, blur=1):
    """
    height, width: original dimensions, before blurring.
    Gaussian: (N, 1, filter_size, filter_size, h, w)
        h = math.ceil(height / blur), w = math.ceil(width / blur)
    """
    bsz, _, filter_size, _, h, w = gaussian.shape

    # norm_tensor = torch.ones([bsz, 1, height, width])
    # normalization_feats = torch.autograd.Variable(norm_tensor)
    # normalization_feats = normalization_feats.to(gaussian.device)
    # norm_out = perform_filtering(normalization_feats, gaussian=gaussian, blur=blur, normalize=False, verbose=False)

    norm_out = torch.ones([bsz, 1, height, width])
    norm_out = torch.autograd.Variable(norm_out)
    norm_out = norm_out.to(gaussian.device)
    norm_out = perform_filtering(norm_out, gaussian=gaussian, blur=blur, normalize=False, verbose=False)

    norm_out = torch.sqrt(norm_out)
    return norm_out