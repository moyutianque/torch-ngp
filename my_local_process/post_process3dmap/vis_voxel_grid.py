import open3d as o3d

filename = './sem_map-insid.ply'
# filename = './sem_map-insrgb.ply'
filename = 'sem_map-insdistortion.ply'
# filename = 'sem_map-insdistortion-cc3d.ply'

voxel_grid = o3d.io.read_voxel_grid(filename)
o3d.visualization.draw_geometries([voxel_grid], width=700, height=700)