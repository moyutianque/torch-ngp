from torch_efficient_distloss import eff_distloss
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import numpy as np

fig =  go.Figure()

def plot_vec(w):
    w_np = w.detach().cpu().numpy()
    fig.add_trace(
        go.Scatter(x=np.arange(len(w_np[0])), y=w_np[0])
    )

def dist_loss(w, t):
    # transfered from jax code of mip nerf 360
    ut = (t[..., 1:] + t[..., :-1]) / 2
    dut = torch.abs(ut[..., :, None] - ut[..., None, :])
    loss_inter = torch.sum(w * torch.sum(w[..., None, :] * dut, dim=-1), dim=-1)
    loss_intra = torch.sum(w**2 * (t[..., 1:] - t[..., :-1]), dim=-1) / 3

    return loss_inter + loss_intra

relu = nn.ReLU()

# Toy example 1D
B = 1
N = 128         # number of points on a ray
feat_dim = 1  # number of instance classes predicted at one point
interval = 1/N
iters = 1000

# w = torch.rand(B, N).cuda()

# single peak
w = [i**2 for i in range(0, N//2, 1)] + [i**2 for i in range(N//2-1, -1, -1)] 

# double peaks
# w = [i**2 for i in range(0, N//4, 1)] + [i**2 for i in range(N//4-1, -1, -1)] + [i**2 for i in range(0, N//4, 1)] + [i**2 for i in range(N//4-1, -1, -1)] 

# double peaks with shift
# w = [0,0,0,0,0] + [i**2 for i in range(0, N//5, 1)] + [i**2 for i in range(N//5-1, -1, -1)] + [i**2 for i in range(0, N//5, 1)] + [i**2 for i in range(N//5-1, -1, -1)] 
# w = w + [0] * (N-len(w))

# test uniform value [cannot keep the unified distribution]
# w = [0.1] * N 

# similar to mip 360 figure 6 input
w = [0.1] * 15 + [0.22] * 5 + [0.01] * 25 + [0.25] * 15 + [0.01] * 25 + [0.23] * 8 + [0.05] * 35
print(len(w))
# w = torch.normal(0, 1, (B,N)).cuda()

w = torch.tensor(w).float().cuda()[None, ...]
w = relu(w)
w = w / w.sum(-1, keepdim=True)
w = w.detach().clone().requires_grad_()
plot_vec(w)

s = torch.linspace(0, 1, N+1).cuda() # not necessary
m = (s[1:] + s[:-1]) * 0.5
m = m[None].repeat(B, 1)
s = s[None].repeat(B, 1)

for i in range(iters):
    if i > 0:
        w = relu(w) # if w has negative, the optimization does not have meaning
        w = w / w.sum(-1, keepdim=True)
        w = w.detach().clone().requires_grad_()
    optimizer = torch.optim.Adam([w], lr=0.01)
    optimizer.zero_grad()
    
    # loss = dist_loss(w, s) # has checked the equivlence 
    loss = eff_distloss(w, m, interval)
    loss.backward()
    optimizer.step()

    if i%50 == 0:
        plot_vec(relu(w))

    print('Loss', loss)

# visualization
steps = []
for i in range(len(fig.data)):
    step = dict(
        method="update",
        args=[{"visible": [False] * len(fig.data)},
              {"title": "Slider switched to step: " + str(i)}],  # layout attribute
    )
    step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
    steps.append(step)

sliders = [dict(
    active=0,
    currentvalue={"prefix": "Optimization step: "},
    pad={"t": 50},
    steps=steps
)]

fig.update_layout(
    sliders=sliders
)

fig.show()
fig.write_html("out.html")

