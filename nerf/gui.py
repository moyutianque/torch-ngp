import math
import torch
import numpy as np
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R

from .utils import *
from PIL import Image
d3_40_colors_rgb: np.ndarray = np.array(
    [
        [1, 1, 1],
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [214, 39, 40],
        [255, 152, 150],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [206, 219, 156],
        [140, 109, 49],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)

class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        # self.radius = r # camera distance from center
        self.radius = 0 # NOTE zehao modified: rotate along frame center
        self.fovy = fovy # in degree
        # self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.center = np.array([0, 0, -r], dtype=np.float32) # # NOTE zehao modified camera shift

        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.1 * dx)
        rotvec_y = side * np.radians(-0.1 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot
    
    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])
    

class NeRFGUI:
    def __init__(self, opt, trainer, train_loader=None, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.debug = debug
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg
        self.training = False
        self.step = 0 # training step 
        self.global_iter = 0

        self.map_res = 512

        self.trainer = trainer
        self.train_loader = train_loader
        
        self.val_data = [
            train_loader._data.poses_verify, 
            train_loader._data.images_verify,
            train_loader._data.sem_datas_verify,
        ]

        if train_loader is not None:
            self.trainer.error_map = train_loader._data.error_map

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        self.spp = 1 # sample per pixel
        self.mode = 'image' # choose from ['image', 'depth']

        self.dynamic_resolution = True
        self.downscale = 1
        self.train_steps = 16

        dpg.create_context()
        self.register_dpg()
        self.test_step()
        

    def __del__(self):
        dpg.destroy_context()


    def train_step(self):

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        starter.record()

        outputs = self.trainer.train_gui(self.train_loader, step=self.train_steps, val_data=self.val_data, iters=self.global_iter)

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        self.step += self.train_steps
        self.global_iter += 1
        self.need_update = True

        dpg.set_value("_log_train_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
        if "l_sem" in outputs.keys():
            dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), lrgb = {outputs["l_rgb"]:.4f}, lsem = {outputs["l_sem"]:.4f}, lds = {outputs["l_ds"]:.4f}, lr = {outputs["lr"]:.5f}')
        else:
            dpg.set_value("_log_train_log", f'step = {self.step: 5d} (+{self.train_steps: 2d}), loss = {outputs["loss"]:.4f}, lr = {outputs["lr"]:.5f}')

        # dynamic train steps
        # max allowed train time per-frame is 500 ms
        full_t = t / self.train_steps * 16
        train_steps = min(16, max(4, int(16 * 500 / full_t)))
        if train_steps > self.train_steps * 1.2 or train_steps < self.train_steps * 0.8:
            self.train_steps = train_steps

    def prepare_buffer(self, outputs):
        if self.mode == 'image':
            return outputs['image']
        elif self.mode == 'sem':
            if "rgb" in self.opt.sem_mode:
                return outputs['sem']
            else:
                outputs['sem'] = np.argmax(outputs['sem'], axis=-1).astype(int)
                out_sem = d3_40_colors_rgb[outputs['sem'] % 40].astype(np.float32)/255.
                return out_sem
        else:
            return np.expand_dims(outputs['depth'], -1).repeat(3, -1)

    
    def test_step(self):
        # TODO: seems we have to move data from GPU --> CPU --> GPU?

        if self.need_update or self.spp < self.opt.max_spp:
        
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()

            outputs = self.trainer.test_gui(self.cam.pose, self.cam.intrinsics, self.W, self.H, self.bg_color, self.spp, self.downscale)

            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)

            # update dynamic resolution
            if self.dynamic_resolution:
                # max allowed infer time per-frame is 200 ms
                full_t = t / (self.downscale ** 2)
                downscale = min(1, max(1/4, math.sqrt(200 / full_t)))
                if downscale > self.downscale * 1.2 or downscale < self.downscale * 0.8:
                    self.downscale = downscale

            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.spp = 1
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
                self.spp += 1

            dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            dpg.set_value("_log_resolution", f'{int(self.downscale * self.W)}x{int(self.downscale * self.H)}')
            dpg.set_value("_log_spp", self.spp)
            if os.environ.get('DEBUG', False): 
                print(self.render_buffer.shape)
            dpg.set_value("_texture", self.render_buffer)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=500, height=300):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # time
            if not self.opt.test:
                with dpg.group(horizontal=True):
                    dpg.add_text("Train time: ")
                    dpg.add_text("no data", tag="_log_train_time")                    

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            with dpg.group(horizontal=True):
                dpg.add_text("SPP: ")
                dpg.add_text("1", tag="_log_spp")

            # train button
            if not self.opt.test:
                with dpg.collapsing_header(label="Train", default_open=True):

                    # train / stop
                    with dpg.group(horizontal=True):
                        dpg.add_text("Train: ")

                        def callback_train(sender, app_data):
                            if self.training:
                                self.training = False
                                dpg.configure_item("_button_train", label="start")
                            else:
                                self.training = True
                                dpg.configure_item("_button_train", label="stop")

                        dpg.add_button(label="start", tag="_button_train", callback=callback_train)
                        dpg.bind_item_theme("_button_train", theme_button)

                        def callback_reset(sender, app_data):
                            @torch.no_grad()
                            def weight_reset(m: nn.Module):
                                reset_parameters = getattr(m, "reset_parameters", None)
                                if callable(reset_parameters):
                                    m.reset_parameters()
                            self.trainer.model.apply(fn=weight_reset)
                            self.trainer.model.reset_extra_state() # for cuda_ray density_grid and step_counter
                            self.need_update = True

                        dpg.add_button(label="reset", tag="_button_reset", callback=callback_reset)
                        dpg.bind_item_theme("_button_reset", theme_button)

                    # save ckpt
                    with dpg.group(horizontal=True):
                        dpg.add_text("Checkpoint: ")

                        def callback_save(sender, app_data):
                            self.trainer.save_checkpoint(full=True, best=False)
                            dpg.set_value("_log_ckpt", "saved " + os.path.basename(self.trainer.stats["checkpoints"][-1]))
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="save", tag="_button_save", callback=callback_save)
                        dpg.bind_item_theme("_button_save", theme_button)

                        dpg.add_text("", tag="_log_ckpt")
                    
                    # save mesh
                    with dpg.group(horizontal=True):
                        dpg.add_text("Marching Cubes: ")

                        def callback_mesh(sender, app_data):
                            self.trainer.save_mesh(resolution=256, threshold=10)
                            dpg.set_value("_log_mesh", "saved " + f'{self.trainer.name}_{self.trainer.epoch}.ply')
                            self.trainer.epoch += 1 # use epoch to indicate different calls.

                        dpg.add_button(label="mesh", tag="_button_mesh", callback=callback_mesh)
                        dpg.bind_item_theme("_button_mesh", theme_button)

                        dpg.add_text("", tag="_log_mesh")
                    
                    with dpg.group(horizontal=True):
                        dpg.add_text("voxelized map: ")

                        def callback_save_map(sender, app_data):
                            print("Start dumping 3d map")
                            self.trainer.save_3dmap(resolution=self.map_res)
                            print("Finish dumping 3d map")
                            dpg.set_value("_log_3dmap", "saved " + f'{self.trainer.name}_3dmap.npy')

                        dpg.add_button(label="save_map", tag="_button_3dmap", callback=callback_save_map)
                        dpg.bind_item_theme("_button_3dmap", theme_button)

                        dpg.add_text("", tag="_log_3dmap")
                        
                    with dpg.group(horizontal=True):    
                        def callback_set_map_res(sender, app_data):
                            self.map_res = int(app_data)
                            self.need_update = True

                        dpg.add_slider_int(label="Map resolution", min_value=256, max_value=2048, format="%d res.", default_value=self.map_res, callback=callback_set_map_res)

                    with dpg.group(horizontal=True):
                        dpg.add_text("", tag="_log_train_log")

            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # dynamic rendering resolution
                with dpg.group(horizontal=True):

                    def callback_set_dynamic_resolution(sender, app_data):
                        if self.dynamic_resolution:
                            self.dynamic_resolution = False
                            self.downscale = 1
                        else:
                            self.dynamic_resolution = True
                        self.need_update = True

                    dpg.add_checkbox(label="dynamic resolution", default_value=self.dynamic_resolution, callback=callback_set_dynamic_resolution)
                    dpg.add_text(f"{self.W}x{self.H}", tag="_log_resolution")

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('image', 'depth', 'sem'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # bg_color picker
                def callback_change_bg(sender, app_data):
                    self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                    self.need_update = True

                dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)

                # dt_gamma slider
                def callback_set_dt_gamma(sender, app_data):
                    self.opt.dt_gamma = app_data
                    self.need_update = True

                dpg.add_slider_float(label="dt_gamma", min_value=0, max_value=0.1, format="%.5f", default_value=self.opt.dt_gamma, callback=callback_set_dt_gamma)

                # max_steps slider
                def callback_set_max_steps(sender, app_data):
                    self.opt.max_steps = app_data
                    self.need_update = True

                dpg.add_slider_int(label="max steps", min_value=1, max_value=1024, format="%d", default_value=self.opt.max_steps, callback=callback_set_max_steps)

                # aabb slider
                def callback_set_aabb(sender, app_data, user_data):
                    # user_data is the dimension for aabb (xmin, ymin, zmin, xmax, ymax, zmax)
                    self.trainer.model.aabb_infer[user_data] = app_data

                    # also change train aabb ? [better not...]
                    #self.trainer.model.aabb_train[user_data] = app_data

                    self.need_update = True

                dpg.add_separator()
                dpg.add_text("Axis-aligned bounding box:")

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="x", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=0)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=3)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="y", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=1)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=4)

                with dpg.group(horizontal=True):
                    dpg.add_slider_float(label="z", width=150, min_value=-self.opt.bound, max_value=0, format="%.2f", default_value=-self.opt.bound, callback=callback_set_aabb, user_data=2)
                    dpg.add_slider_float(label="", width=150, min_value=0, max_value=self.opt.bound, format="%.2f", default_value=self.opt.bound, callback=callback_set_aabb, user_data=5)
                

            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))

        def callback_camera_move(sender, app_data):
            """ 
                wasd (87,65,83,68) move 
                ijkl(73,74,75,76) rotate, 
                eq (69,81) up, down
                cz(67,90) zoom in/out
            """
            if not dpg.is_item_focused("_primary_window"):
                return
            if app_data in {87, 65, 83, 68, 73, 74, 75, 76, 69, 81, 67, 90}:
                # move
                if app_data == 87: # w
                    self.cam.pan(0, 0, -200)
                elif app_data == 83: # s
                    self.cam.pan(0, 0, 200)
                elif app_data==65: # a
                    self.cam.pan(100, 0)
                elif app_data==68: # d
                    self.cam.pan(-100, 0)
                # rotate
                elif app_data == 73: # i
                    self.cam.orbit(0, -50)
                elif app_data == 75: # k
                    self.cam.orbit(0, 50)
                elif app_data==74: # j
                    self.cam.orbit(-50, 0)
                elif app_data==76: # l
                    self.cam.orbit(50, 0)
                # up down
                elif app_data == 69: # e
                    self.cam.pan(0, 100)
                elif app_data == 81: # q
                    self.cam.pan(0, -100)
                # zoom in out
                elif app_data == 67: #c
                    self.cam.scale(1)
                elif app_data == 90: # z
                    self.cam.scale(-1)
                self.need_update = True


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)
            
            dpg.add_key_press_handler(callback=callback_camera_move)
        
        dpg.create_viewport(title='torch-ngp', width=self.W, height=self.H, resizable=False)
        
        # TODO: seems dearpygui doesn't support resizing texture...
        # def callback_resize(sender, app_data):
        #     self.W = app_data[0]
        #     self.H = app_data[1]
        #     # how to reload texture ???

        # dpg.set_viewport_resize_callback(callback_resize)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self): 
        while dpg.is_dearpygui_running():
            # update texture every frame
            if self.training:
                self.train_step()
                # NOTE: zehao temporally add novel view synthesis in gui mode
                # if self.step > self.opt.iters:
                #     self.test_step()
                #     dpg.render_dearpygui_frame()
                #     break
            self.test_step()
            dpg.render_dearpygui_frame()