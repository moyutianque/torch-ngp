scene_path=data/replica_apart2_instance
python main_nerf.py $scene_path \
--workspace outputs/replica_apart2_instance --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --gui --W 1440 --H 900 --iters 30000 --save_iter 600 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 6144 #--scale 1 --bound 1 
