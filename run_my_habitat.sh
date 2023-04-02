scene_path=data/replica_multi_room 
scene_path=data/replica_multi_room_instance
python main_nerf.py $scene_path \
--workspace outputs/replica_multi_room --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --gui --W 1440 --H 900 --iters 10000 \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 #--scale 1 --bound 1