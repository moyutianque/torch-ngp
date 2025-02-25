import torch
import argparse

from nerf_sem.provider import NeRFDataset
from nerf.gui import NeRFGUI
from nerf_sem.utils import *

from easydict import EasyDict as edict
from diffusers import StableDiffusionImg2ImgPipeline
#torch.autograd.set_detect_anomaly(True)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('-O', action='store_true', help="equals --fp16 --cuda_ray --preload")
    parser.add_argument('--test', action='store_true', help="test mode")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--iters', type=int, default=30000, help="training iters")
    parser.add_argument('--lr', type=float, default=1e-2, help="initial learning rate")
    parser.add_argument('--lr_latent', type=float, default=1e-4, help="initial learning rate")
    parser.add_argument('--ckpt', type=str, default='latest')
    parser.add_argument('--num_rays', type=int, default=4096, help="num rays sampled per image for each training step")
    parser.add_argument('--cuda_ray', action='store_true', help="use CUDA raymarching instead of pytorch")
    parser.add_argument('--max_steps', type=int, default=1024, help="max num steps sampled per ray (only valid when using --cuda_ray)")
    parser.add_argument('--num_steps', type=int, default=512, help="num steps sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--upsample_steps', type=int, default=0, help="num steps up-sampled per ray (only valid when NOT using --cuda_ray)")
    parser.add_argument('--update_extra_interval', type=int, default=16, help="iter interval to update extra status (only valid when using --cuda_ray)")
    parser.add_argument('--max_ray_batch', type=int, default=4096, help="batch size of rays at inference to avoid OOM (only valid when NOT using --cuda_ray)")
    parser.add_argument('--patch_size', type=int, default=1, help="[experimental] render patches in training, so as to apply LPIPS loss. 1 means disabled, use [64, 32, 16] to enable")

    ### network backbone options
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")
    parser.add_argument('--tcnn_sem', action='store_true', help="use TCNN semantic backend")

    ### dataset options
    parser.add_argument('--color_space', type=str, default='srgb', help="Color space, supports (linear, srgb)")
    parser.add_argument('--preload', action='store_true', help="preload all data into GPU, accelerate training but use more GPU memory")
    # (the default value is for the fox dataset)
    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--scale', type=float, default=0.33, help="scale camera location into box[-bound, bound]^3")
    parser.add_argument('--offset', type=float, nargs='*', default=[0, 0, 0], help="offset of camera location")
    parser.add_argument('--dt_gamma', type=float, default=1/128, help="dt_gamma (>=0) for adaptive ray marching. set to 0 to disable, >0 to accelerate rendering (but usually with worse quality)")
    parser.add_argument('--min_near', type=float, default=0.2, help="minimum near distance for camera")
    parser.add_argument('--density_thresh', type=float, default=10, help="threshold for density grid to be occupied")
    parser.add_argument('--bg_radius', type=float, default=-1, help="if positive, use a background model at sphere(bg_radius)")

    ### GUI options
    parser.add_argument('--gui', action='store_true', help="start a GUI")
    parser.add_argument('--W', type=int, default=1920, help="GUI width")
    parser.add_argument('--H', type=int, default=1080, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    parser.add_argument('--max_spp', type=int, default=64, help="GUI rendering max sample per pixel")

    ### experimental
    parser.add_argument('--error_map', action='store_true', help="use error map to sample rays")
    parser.add_argument('--clip_text', type=str, default='', help="text input for CLIP guidance")
    parser.add_argument('--rand_pose', type=int, default=-1, help="<0 uses no rand pose, =0 only uses rand pose, >0 sample one rand pose every $ known poses")

    ### GUI aabb bounding box for rendering
    parser.add_argument('--bx', type=float, default=-2)
    parser.add_argument('--by', type=float, default=-2)
    parser.add_argument('--bz', type=float, default=-2)
    parser.add_argument('--tx', type=float, default=2)
    parser.add_argument('--ty', type=float, default=2)
    parser.add_argument('--tz', type=float, default=2)

    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--warmup_iter', type=int, default=10000)
    parser.add_argument('--dist_start', type=int, default=15000)
    parser.add_argument('--distortion_loss', action='store_true', help="distortion loss of mip nerf 360")
    parser.add_argument('--depth_reg', action='store_true', help="depth regularization loss of regnerf")
    parser.add_argument('--split_sem_code', action='store_true', help="")
    parser.add_argument('--sigma_dropout', type=int, default=0, help="sigma dropout")
    parser.add_argument('--depth_sup', action='store_true', help="include depth supervision")
    parser.add_argument('--post_3dmap_loss', action='store_true', help="include depth supervision")
    parser.add_argument('--density_sample_size', type=int, default=2000, help="sigma dropout")
    parser.add_argument('--radial_depth', action='store_true', help="change plane-to-plane depth to radial length")
    parser.add_argument('--reprojection_loss', action='store_true', help="reprojection")
    parser.add_argument('--latent', action='store_true', help="latent nerf")
    parser.add_argument('--low_res_img', action='store_true', help="latent nerf with low res image")

    parser.add_argument('--load_sem', action='store_true', help="load semantic annotation in data loader")
    # NOTE: add normal estimation task
    parser.add_argument('--use_normal', action='store_true', help="add normal estimation problem")
    parser.add_argument('--sem_label', action='store_true', help="whether use in normal prediction")
    parser.add_argument('--sem_ins', action='store_true', help="whether use in normal prediction")

    parser.add_argument('--use_depth', action='store_true', help="add depth estimation problem")
    opt = parser.parse_args()

    if opt.O:
        opt.fp16 = True
        opt.cuda_ray = True
        opt.preload = True
    
    if opt.patch_size > 1:
        opt.error_map = False # do not use error_map if use patch-based training
        # assert opt.patch_size > 16, "patch_size should > 16 to run LPIPS loss."
        assert opt.num_rays % (opt.patch_size ** 2) == 0, "patch_size ** 2 should be dividable by num_rays."


    if opt.ff:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --ff"
        from nerf.network_ff import NeRFNetwork
    elif opt.tcnn:
        opt.fp16 = True
        assert opt.bg_radius <= 0, "background model is not implemented for --tcnn"
        if opt.tcnn_sem:
            if opt.split_sem_code:
                from nerf_sem.network_tcnn_insid_split_semcode import NeRFNetwork
            else:
                from nerf_sem.network_tcnn_insid import NeRFNetwork
               
        else:
            from nerf.network_tcnn import NeRFNetwork
    else:
        from nerf.network import NeRFNetwork

    print(opt)
    
    seed_everything(opt.seed)


    criterion = torch.nn.MSELoss(reduction='none')
    criterion_ce = torch.nn.CrossEntropyLoss(reduction='none')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vae=None
    if opt.latent:
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1", 
            use_auth_token=True
        )
        vae = pipe.vae
        vae = vae.to(device)
        vae = vae.eval()
    
    if opt.test:
        
        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        
        # TODO currently not work for sem, because not modified
        model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            aabb_bounds=[opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz],
        )

        trainer = Trainer('ngp', opt, model, device=device, workspace=opt.workspace, criterion=criterion, criterion_sem=criterion_sem, fp16=opt.fp16, metrics=metrics, use_checkpoint=opt.ckpt)

        if opt.gui:
            gui = NeRFGUI(opt, trainer)
            gui.render()
        
        else:
            test_loader = NeRFDataset(opt, device=device, type='test', vae=vae).dataloader()

            if test_loader.has_gt:
                trainer.evaluate(test_loader) # blender has gt, so evaluate it.
    
            trainer.test(test_loader, write_video=True) # test and save video
            
            trainer.save_mesh(resolution=256, threshold=10)
    
    else:
        train_loader = NeRFDataset(opt, device=device, type='train', vae=vae).dataloader()
        extra_configs = [edict({'dim_out':32, 'hidden_dim': 64, 'num_layers':2, 'geo_only': False, 'act_type': None})]
        extra_configs = []

        model = NeRFNetwork(
            encoding="hashgrid",
            bound=opt.bound,
            cuda_ray=opt.cuda_ray,
            density_scale=1,
            min_near=opt.min_near,
            density_thresh=opt.density_thresh,
            bg_radius=opt.bg_radius,
            aabb_bounds=[opt.bx, opt.by, opt.bz, opt.tx, opt.ty, opt.tz],
            sem_label_emb = max(train_loader._data.target_labels_sem),
            sem_ins_emb = max(train_loader._data.target_labels_ins),
            extra_configs=extra_configs
        )
        
        print(model)

        optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15)
        # optimizer = lambda model: torch.optim.Adam(model.get_params(opt.lr), betas=(0.9, 0.99), eps=1e-15, weight_decay=1e-5) # NOTE zehao add L2 normalization on weights

        # decay to 0.1 * init_lr at last iter step
        scheduler = lambda optimizer: optim.lr_scheduler.LambdaLR(optimizer, lambda iter: 0.1 ** min(iter / opt.iters, 1))

        metrics = [PSNRMeter(), LPIPSMeter(device=device)]
        trainer = Trainer(
            'ngp', opt, model, device=device, workspace=opt.workspace, 
            optimizer=optimizer, criterion=criterion, criterion_ce=criterion_ce, 
            ema_decay=0.95, fp16=opt.fp16, lr_scheduler=scheduler, 
            scheduler_update_every_step=True, metrics=metrics, 
            use_checkpoint=opt.ckpt, eval_interval=50, save_iter=opt.save_iter, 
            vae=vae, extra_configs=extra_configs
        )

        if opt.gui:
            gui = NeRFGUI(opt, trainer, train_loader)
            gui.render()
        else:
            val_data = [
                train_loader._data.poses_verify, 
                train_loader._data.images_verify,
                train_loader._data.depths_datas_verify,
                train_loader._data.extra_inputs_verify,
            ]
            global_iter = 0
            while global_iter < opt.iters:
                outputs = trainer.train(train_loader, step=16, val_data=val_data)
                global_iter = trainer.global_step
                print("Step: ", global_iter)
                for k,v in outputs.items():
                    print(k, ":", v)
                print()
            print("Experiment finished")
