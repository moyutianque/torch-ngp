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
                 use_latent=False,
                 low_res_img=False,
                 latent_dim=32,
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

        freq = 12
        print(f"Using frequency position encoding with freq {freq}")
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

        self.use_latent = use_latent
        self.low_res_img = low_res_img

        self.latent_dim = latent_dim
        self.latent_proj = nn.Linear(self.latent_dim, 4)
        
        if use_latent:
            self.color_net = tcnn.Network(
                n_input_dims=self.in_dim_color,
                n_output_dims=self.latent_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": hidden_dim_color,
                    "n_hidden_layers": num_layers_color - 1,
                },
            )

            if self.low_res_img:
                self.color_net_low_res = tcnn.Network(
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


        else:
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
            if os.environ.get('ONLYCOORD', False):
                if os.environ.get('TABNET', False):
                    # self.sem_net = tab_network.TabNet(
                    #     freq*3*2,
                    #     sem_out_dim,
                    #     n_d=8,
                    #     n_a=8,
                    #     n_steps=3,
                    #     gamma=1.3,
                    #     cat_idxs=[],
                    #     cat_dims=[],
                    #     cat_emb_dim=1,
                    #     n_independent=2,
                    #     n_shared=2,
                    #     epsilon=1e-15,
                    #     virtual_batch_size=128,
                    #     momentum=0.02,
                    #     mask_type="sparsemax",
                    # )
                    # self.sem_net = GradientBoostingClassifier(n_estimators=100, learning_rate=1e-3, max_depth=5)
                    raise NotImplementedError()
                else:
                    self.sem_net = tcnn.Network(
                        n_input_dims=freq*3*2,
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
                    n_input_dims=self.geo_feat_dim + freq*3*2,
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

        # self.relu = nn.ReLU()
        self.softplus = nn.Softplus()
    
    def sem_activation(self, x):
        # return torch.nn.functional.softmax(x, dim=-1)
        # return torch.sigmoid(x)
        # return x
        return self.softplus(x)
        # return self.relu(x)
       
    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        if os.environ.get('BLOCKSEM', False):
            x_sincos = self.sincos_encoder(x)

        x = self.encoder(x)
        h = self.sigma_net(x)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h_cat = torch.cat([d, geo_feat], dim=-1)
        if self.use_latent and self.low_res_img:
            h_im = self.color_net_low_res(h_cat)
        h = self.color_net(h_cat)

        # NOTE zehao@nov 28 
        if os.environ.get('BLOCKSEM', False):
            if os.environ.get('ONLYCOORD', False):
                geo_feat2 = x_sincos.detach()
            else:
                geo_feat2 = torch.cat([x_sincos.detach(), geo_feat.detach()], dim=-1)
            # geo_feat2 = geo_feat.detach()
            
            if os.environ.get('TABNET', False):
                try:
                    h_sem, M_LOSS = self.sem_net(geo_feat2)
                    
                except:
                    import ipdb;ipdb.set_trace() # breakpoint 202
                # import ipdb;ipdb.set_trace() # breakpoint 197
            else:
                h_sem = self.sem_net(geo_feat2)
        else:
            h_sem = self.sem_net(geo_feat)
        
        # sigmoid activation for rgb
        if self.use_latent:
            color_low_res = None
            if self.low_res_img:
                color_low_res = torch.sigmoid(h_im)
            color=h
            return sigma, color, color_low_res
        
        color = torch.sigmoid(h)
        sem = self.sem_activation(h_sem)
        return sigma, color, sem

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

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        import ipdb;ipdb.set_trace() # breakpoint 278
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
        if self.use_latent:
            h = h
        else:
            h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs     

    def sem(self, x, mask=None, geo_feat=None, **kwargs):
        import ipdb;ipdb.set_trace() # breakpoint 211
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        if os.environ.get('BLOCKSEM', False):
            x_sincos = self.sincos_encoder(x)

        if mask is not None:
            sems = torch.zeros(mask.shape[0], self.sem_out_dim, dtype=x.dtype, device=x.device) # [N, sem_out_dim]
            # in case of empty mask
            if not mask.any():
                return sems
            x = x[mask]
            geo_feat = geo_feat[mask]

        # sem
        if os.environ.get('BLOCKSEM', False):
            geo_feat2 = torch.cat([x_sincos.detach(), geo_feat.detach()], dim=-1)
            # geo_feat2 = geo_feat.detach()
            h_sem = self.sem_net(geo_feat2)
            h = self.sem_net(h_sem)
        else:
            h = self.sem_net(geo_feat)

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
            {'params': self.latent_proj.parameters(), 'lr': lr}
        ]
        if self.use_latent and self.low_res_img:
            params.append({'params': self.color_net_low_res.parameters(), 'lr': lr})
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params