import numpy as np
from scipy.ndimage import maximum_filter
from VoxelData import VoxelData
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import argparse

def draw_map(m, mask, downsample_scale = 4, dump_path="sem_map.html"):
    m = m.astype(int)
    sem_labels = np.unique(m[mask])

    m = maximum_filter(m, size=downsample_scale, mode='constant')
    m = m[::downsample_scale, ::downsample_scale, ::downsample_scale]
    # Visualize

    mapper = {l:i for i, l in enumerate(sorted(sem_labels))}
    import ipdb;ipdb.set_trace() # breakpoint 16

    color_palette = plt.get_cmap('Spectral')(np.linspace(0, 1, len(mapper)))

    m = np.vectorize(mapper.get)(m)
    facecolors = color_palette[m]

    Voxels = VoxelData(m>0, facecolors)  # show only navigable area

    color = np.array(Voxels.colors)
    color[:, :3] *= 255
    color = [f"rgba({int(rgba[0])}, {int(rgba[1])}, {int(rgba[2])}, {1})" for rgba in color]
    data = []
    data.append(
        go.Mesh3d(
            x=Voxels.vertices[0],
            y=Voxels.vertices[1],
            z=Voxels.vertices[2],
            i=Voxels.triangles[0],
            j=Voxels.triangles[1],
            k=Voxels.triangles[2],
            facecolor=color,
            opacity=1.,
        )
    )

    layout = go.Layout(
                scene=dict(
                    aspectmode='data'
            )) 

    fig = go.Figure(data=data,layout=layout)
    fig.write_html(dump_path)

def load_map(map_path):
    map_np = np.load(map_path, allow_pickle=True)[()]
    return map_np['density'], map_np['sem']

def filter_map(m,bx,by,bz,tx,ty,tz):
    return m[bx:tx, by:ty, bz:tz]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bx', type=int, default=0)
    parser.add_argument('--by', type=int, default=0)
    parser.add_argument('--bz', type=int, default=0)
    parser.add_argument('--tx', type=int, default=-1)
    parser.add_argument('--ty', type=int, default=-1)
    parser.add_argument('--tz', type=int, default=-1)
    parser.add_argument('--d_thresh', type=float, default=1)
    parser.add_argument('--map_path', type=str, default='../../outputs/replica_apart2_dinning_room/ngp_3dmap-512.npy')
    opt = parser.parse_args()

    dm, sm = load_map(opt.map_path)
    dm = filter_map(dm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)
    sm = filter_map(sm, opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz)

    mask = dm > opt.d_thresh
    sm[~mask] = 0
    draw_map(sm, mask, downsample_scale=4)


