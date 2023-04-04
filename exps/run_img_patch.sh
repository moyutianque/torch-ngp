# scene_path=data/replica_multiroom2
# output_path=replica_patch
# python main_nerf.py $scene_path \
# --workspace outputs/$output_path --fp16 --tcnn --preload --cuda_ray \
# --tcnn_sem --iters 30000 --save_iter 5000 --min_near 0. \
# --bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 8192 \
# --warmup_iter 50000 --patch_size 32


scene_path=data/replica_dinning_room
output_path=replica_patch
DEBUG_fixed_decoder=1 python main_nerf.py $scene_path \
--workspace outputs/$output_path --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --iters 100000 --save_iter 10000 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 10240 \
--warmup_iter 100000 --patch_size 32 --latent_dim 32