# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 15:47:55 2023
@author: zhangxin
"""

from .modules.seanet import SEANetEncoder, SEANetDecoder
from .quantizer import ResidualVectorQuantize
import torch.nn as nn
from einops import rearrange
import torch
import numpy as np
import math


class SASCodec(nn.Module):
    def __init__(
            self, 
            n_filters=64,
            strides=[8,5,4,2],
            dimension=1024,
            semantic_dimension=1024,
            lstm_layers=2,
            residual_kernel_size=3,
            dilation_base=2,
            n_residual_layers=1,
            n_codebooks=16,
            codebook_size=1024,
            codebook_dim=16,
            quantizer_dropout=0.5,
            activation='ELU',
            bidirectional=True,
            sample_rate=16000,
            **kwargs,
            ):
        '''
        
        Parameters
        ----------
        config : json
            Model Config.

        '''
        super().__init__()
        self.encoder = SEANetEncoder(
            n_filters=n_filters, 
            dimension=dimension, 
            ratios=strides,
            lstm=lstm_layers,
            bidirectional=bidirectional,
            dilation_base=dilation_base,
            residual_kernel_size=residual_kernel_size,
            n_residual_layers=n_residual_layers,
            activation=activation
            )
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size
        self.n_codebooks = n_codebooks
        self.downsample_rate = np.prod(strides)
        self.transform = nn.Conv1d(dimension, semantic_dimension, 1)

        self.quantizer = ResidualVectorQuantize(
            input_dim=dimension, 
            n_codebooks=n_codebooks, 
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            quantizer_dropout=quantizer_dropout,
            )
        self.decoder = SEANetDecoder(
            n_filters=n_filters, 
            dimension=dimension, 
            ratios=strides,
            lstm=lstm_layers,
            bidirectional=False,
            dilation_base=dilation_base,
            residual_kernel_size=residual_kernel_size,
            n_residual_layers=n_residual_layers,
            activation=activation
            )
        self.hop_length = np.prod(strides)
        
    def preprocess(self, audio_data):

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data
    
    
    def forward(self, 
                x: torch.tensor, 
                n_q: int=None, 
                semantic: bool=True,):
        '''
        
        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.
        layers : list[int], optional
            Layers of RVQ should return quantized result. The default is the first layer.

        Returns
        -------
        o : torch.tensor
            Output wavs. Shape: (batch, channels, timesteps).
        commit_loss : torch.tensor
            Commitment loss from residual vector quantizers.
        feature : torch.tensor
            Output of RVQ's first layer. Shape: (batch, timesteps, dimension)

        '''
        x = self.preprocess(x)
        n_q = n_q if n_q else self.n_codebooks
        e = self.encoder(x)
        z_q, codes, latents, commit_loss, codebook_loss, z_q_0 = self.quantizer(e, n_quantizers=n_q)
        o = self.decoder(z_q)

        if semantic:
            s_out = self.transform(z_q_0)
            s_out = rearrange(s_out, 'b d t -> b t d') # (B, T, D)

        return {
            'audio': o,
            'commit_loss': commit_loss,
            'codebook_loss': codebook_loss,
            'semantic': s_out,
            'audio_0': self.decoder(z_q_0),
        }
    
    
    def encode(self, 
               x: torch.tensor, 
               n_q: int=None):
        '''

        Parameters
        ----------
        x : torch.tensor
            Input wavs. Shape: (batch, channels, timesteps).
        n_q : int, optional
            Number of quantizers in RVQ used to encode. The default is all layers.

        Returns
        -------
        codes : torch.tensor
            Output indices for each quantizer. Shape: (batch, n_q, timesteps)

        '''
        x = self.preprocess(x)
        e = self.encoder(x)
        n_q = n_q if n_q else self.n_codebooks
        codes, _ = self.quantizer.encode(e, n_q)
        return codes
    
    def decode(self, 
               codes: torch.tensor, 
            ):
        '''

        Parameters
        ----------
        codes : torch.tensor
            Indices for each quantizer. Shape: (batch, n_q, timesteps).
        st : int, optional
            Start quantizer index in RVQ. The default is 0.

        Returns
        -------
        o : torch.tensor
            Reconstruct wavs from codes. Shape: (batch, channels, timesteps)

        '''
        quantized = self.quantizer.decode(codes)
        audio = self.decoder(quantized)
        return audio

    @classmethod
    def from_pretrained(cls, path):
        state_dict = torch.load(path, map_location='cpu')
        model = cls(**state_dict['config'])
        model.load_state_dict(state_dict['model'])
        return model