import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import tinycudann as tcnn
from activation import trunc_exp
from .renderer import NeRFRenderer
import os

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 hidden_dim_sem = 64,
                 num_layers_sem=2,
                 sem_out_dim=3,
                 bound=1,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        per_level_scale = np.exp2(np.log2(2048 * bound / 16) / (16 - 1))

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": per_level_scale,
            },
        )

        n_levels_sem = 8
        per_level_scale_sem = np.exp2(np.log2(2048 * bound / n_levels_sem) / (n_levels_sem - 1))
        self.encoder_sem = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels_sem, # default == 16
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": n_levels_sem,
                "per_level_scale": per_level_scale_sem,
            },
        )

        freq = 12
        self.sincos_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency", 
                "n_frequencies": freq   
            }    
        )

        self.sigma_net = tcnn.Network(
            n_input_dims=32,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )
        if os.environ.get('BLOCKSEM', False):
            self.sem_net = tcnn.Network(
                n_input_dims=n_levels_sem*2 + freq*3*2,
                n_output_dims=sem_out_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_sem,
                    "n_hidden_layers": num_layers_sem - 1,
                },
            )
        else:
            self.sem_net = tcnn.Network(
                n_input_dims=self.geo_feat_dim,
                n_output_dims=sem_out_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_sem,
                    "n_hidden_layers": num_layers_sem - 1,
                },
            )

        self.sem_out_dim = sem_out_dim

        print("\033[92m [INFO] \033[00m Running in the model of separate hash code")
        self.softplus = nn.Softplus()
    
    def sem_activation(self, x):
        # return torch.nn.functional.softmax(x, dim=-1)
        # return torch.sigmoid(x)
        return self.softplus(x)
       
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        if os.environ.get('BLOCKSEM', False):
            x_sincos = self.sincos_encoder(x)

        x_out = self.encoder(x)
        h = self.sigma_net(x_out)
        x_out_sem = self.encoder_sem(x)
        # import ipdb;ipdb.set_trace() # breakpoint 128
        # h_sem = self.sigma_net(x_out_sem)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # NOTE zehao@nov 28 
        if os.environ.get('BLOCKSEM', False):
            geo_feat2 = torch.cat([x_sincos, x_out_sem], dim=-1)
            # h_sem = self.sem_net(x_out_sem)
            h_sem = self.sem_net(geo_feat2)
        else:
            h_sem = self.sem_net(h_sem[..., 1:])

        # sigmoid activation for rgb
        color = torch.sigmoid(h)
        
        sem = self.sem_activation(h_sem)
        # sem = h_sem

        return sigma, color, sem

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_out = self.encoder(x)
        h = self.sigma_net(x_out)

        # x_out_sem = self.encoder_sem(x)
        # h_sem = self.sigma_net(x_out_sem)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            # 'geo_feat_sem': h_sem[..., 1:]
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs     

    def sem(self, x, mask=None, geo_feat_sem=None, **kwargs):
        import ipdb;ipdb.set_trace() # breakpoint 223
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        if mask is not None:
            sems = torch.zeros(mask.shape[0], self.sem_out_dim, dtype=x.dtype, device=x.device) # [N, sem_out_dim]
            # in case of empty mask
            if not mask.any():
                return sems
            x = x[mask]
            geo_feat_sem = geo_feat_sem[mask]

        # sem
        h = self.sem_net(geo_feat_sem)

        h = self.sem_activation(h)
        
        if mask is not None:
            sems[mask] = h.to(sems.dtype) # fp16 --> fp32
        else:
            sems = h

        return sems       

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.sem_net.parameters(), 'lr': lr},  # NOTE zehao @ Dec 13, forget to add previously
            {'params': self.encoder_sem.parameters(), 'lr': lr},  # NOTE zehao @ Jan 24, forget to add previously
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params