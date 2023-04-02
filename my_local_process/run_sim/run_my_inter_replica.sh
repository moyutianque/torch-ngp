scene_path=data/replica_v1/apartment_2/habitat/mesh_semantic.ply 
# scene_path=~/large_data/3D/habitat/scene_datasets/17DRP5sb8fy/17DRP5sb8fy.glb
python viewer_simple.py --scene $scene_path --dataset ./ui_data/mp3d.scene_dataset_config.json
# python viewer.py --scene $scene_path --dataset ./ui_data/mp3d.scene_dataset_config.json

# echo "START transforming to nerf used data format"
# python ../convert_poses.py