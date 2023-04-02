map_path=../../outputs/replica_multiroom/ngp_3dmap-1080-insid-uni_noise.h5
map_path=../../outputs/replica_multiroom/ngp_3dmap-insid.h5

# single room
# python draw_on_open3d.py --map_path $map_path \
# --bx 0.1 --tx 1. --bz 0.1 --tz 0.9 --by 0. --ty 0.6 --d_thresh 100

# multi room
python draw_on_open3d.py --map_path $map_path \
--bx 0. --tx 1. --bz 0. --tz 1. --by 0.3 --ty 0.55 --d_thresh 100 --t1 10 --downsample_scale 4