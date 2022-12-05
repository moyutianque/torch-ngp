scene_path=data/replica_apart2_instance
python main_nerf.py $scene_path \
--workspace outputs/replica_apart2_instance --fp16 --tcnn --preload --cuda_ray \
--tcnn_sem --gui --W 1440 --H 900 --iters 30000 --save_iter 600 --min_near 0. \
--bx -1 --by -1 --bz -1 --tx 1 --ty 1 --tz 1 --num_rays 4096 #--scale 1 --bound 1 


# ffmpeg -i depth.webm -crf 0  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" -r 24 -c:v libx264  depth.mp4

