scene_path=data/replica_multiroom
output_path=replica_multiroom
python main_nerf.py $scene_path \
--workspace outputs/$output_path --fp16 --tcnn --preload \
--tcnn_sem --gui --W 1280 --H 800 --iters 30000 --save_iter 5000 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 1024 \
--sem_mode ins_id --warmup_iter 10000 --distortion_loss --dist_start 15000 --ckpt latest
#--patch_size 16 --depth_reg 

# --scale 1 --bound 1 
