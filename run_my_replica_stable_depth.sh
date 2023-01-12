scene_path=data/replica_multiroom2
output_path=replica_multiroom_stable_depth
python main_nerf.py $scene_path \
--workspace outputs/$output_path --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --gui --W 1280 --H 800 --iters 30000 --save_iter 5000 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 4096 \
--sem_mode ins_id --warmup_iter 100000 --depth_sup  --radial_depth

#--scale 1 --bound 1 
