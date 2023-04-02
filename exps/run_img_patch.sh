scene_path=data/replica_multiroom2
output_path=replica_patch
python main_nerf.py $scene_path \
--workspace outputs/$output_path --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --W 768 --H 768 --iters 30000 --save_iter 5000 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 8192 --patch_size 1 \
--sem_mode label_id --warmup_iter 50000 --patch_size 64 
