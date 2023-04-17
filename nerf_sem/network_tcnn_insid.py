import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import tinycudann as tcnn
from activation import trunc_exp
from .renderer import NeRFRenderer
from pytorch_tabnet import tab_network
from sklearn.ensemble import GradientBoostingClassifier

def get_activation_layer(act_type):
    if act_type is None:
        return None
    elif act_type == 'relu':
        return nn.ReLU()
    elif act_type == 'softplus':
        return nn.Softplus()

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="HashGrid",
                 encoding_dir="SphericalHarmonics",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 bound=1,
                 sem_label_emb=None,
                 sem_ins_emb=None,
                 extra_configs=[],
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

        # freq = 12
        # print(f"Using frequency position encoding with freq {freq}")
        # self.sincos_encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "Frequency", 
        #         "n_frequencies": freq   
        #     }    
        # )

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

        self.extra_nets = nn.ModuleList([])
        self.extra_nets_info = []
        for config in extra_configs:
            geo_only = True
            dim_in = self.geo_feat_dim 
            if not config.geo_only:
                geo_only = False
                dim_in += self.encoder_dir.n_output_dims
            
            self.extra_nets.append(
                    tcnn.Network(
                        n_input_dims=dim_in,
                        n_output_dims=config.dim_out,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "None",
                            "n_neurons": config.hidden_dim,
                            "n_hidden_layers": config.num_layers - 1,
                        },
                    )
            )

            self.extra_nets_info.append([
                get_activation_layer(config.act_type),
                geo_only]
            )

        if sem_label_emb:
            self.sem_label_emb = nn.Embedding(sem_label_emb+10, 16).cuda()

        if sem_ins_emb:
            self.sem_ins_emb = nn.Embedding(sem_ins_emb+10, 16).cuda()
       
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        h_cat = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h_cat)
        color = torch.sigmoid(h)

        extra_outs = []
        # for (layer, (activation, geo_only)) in zip(self.extra_nets, self.extra_nets_info):
        #     if geo_only:
        #         x_tmp = layer(geo_feat)
        #     else:
        #         x_tmp = layer(h_cat)
            
        #     if activation is not None:
        #         x_tmp = activation(x_tmp)
        #     extra_outs.append(x_tmp)
        
        return sigma, color, extra_outs

    def density(self, x):
        # x: [N, 3], in [-bound, bound]
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        } 

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
 
        return params