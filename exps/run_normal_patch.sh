scene_path=data/replica_dinning_room
output_path=replica_patch_normal
DEBUG_fixed_decoder=1 python main_nerf.py $scene_path \
--workspace outputs/$output_path --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --iters 300000 --save_iter 20000 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 10240 \
--warmup_iter 10000 --patch_size 32 \
--load_sem --use_normal --sem_ins --sem_label 