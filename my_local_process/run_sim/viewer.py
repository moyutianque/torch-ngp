# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import ctypes
import math
import os
import string
import sys
import time
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

flags = sys.getdlopenflags()
sys.setdlopenflags(flags | ctypes.RTLD_GLOBAL)

import magnum as mn
import numpy as np
from magnum import shaders, text
from magnum.platform.glfw import Application

import habitat_sim
from habitat_sim import physics
from habitat_sim.logging import LoggingContext, logger
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.utils.settings import default_sim_settings, make_cfg
from PIL import Image
from habitat_sim.utils.common import quat_to_magnum
import numpy as np
import jsonlines
import quaternion
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

RECORD_FREQ = 15

class HabitatSimInteractiveViewer(Application):

    # the maximum number of chars displayable in the app window
    # using the magnum text module. These chars are used to
    # display the CPU/GPU usage data
    MAX_DISPLAY_TEXT_CHARS = 256

    # how much to displace window text relative to the center of the
    # app window (e.g if you want the display text in the top left of
    # the app window, you will displace the text
    # window width * -TEXT_DELTA_FROM_CENTER in the x axis and
    # widnow height * TEXT_DELTA_FROM_CENTER in the y axis, as the text
    # position defaults to the middle of the app window)
    TEXT_DELTA_FROM_CENTER = 0.49

    # font size of the magnum in-window display text that displays
    # CPU and GPU usage info
    DISPLAY_FONT_SIZE = 16.0

    def __init__(self, sim_settings: Dict[str, Any]) -> None:
        self.obs_save_idx = 0
        self.total_frames = 0
        configuration = self.Configuration()
        configuration.title = "Habitat Sim Interactive Viewer"
        Application.__init__(self, configuration)
        self.sim_settings: Dict[str:Any] = sim_settings
        self.fps: float = 60.0

        # draw Bullet debug line visualizations (e.g. collision meshes)
        self.debug_bullet_draw = False
        # draw active contact point debug line visualizations
        self.contact_debug_draw = False
        # cache most recently loaded URDF file for quick-reload
        self.cached_urdf = ""

        # set proper viewport size
        self.viewport_size: mn.Vector2i = mn.gl.default_framebuffer.viewport.size()
        self.sim_settings["width"] = self.viewport_size[0]
        self.sim_settings["height"] = self.viewport_size[1]
        print("Width and Height: ", self.sim_settings["width"], self.sim_settings["height"])

        # set up our movement map
        key = Application.KeyEvent.Key
        self.pressed = {
            key.UP: False,
            key.DOWN: False,
            key.LEFT: False,
            key.RIGHT: False,
            key.A: False,
            key.D: False,
            key.S: False,
            key.W: False,
            key.Q: False,
            key.E: False,
        }

        # set up our movement key bindings map
        key = Application.KeyEvent.Key
        self.key_to_action = {
            key.UP: "look_up",
            key.DOWN: "look_down",
            key.LEFT: "turn_left",
            key.RIGHT: "turn_right",
            key.A: "move_left",
            key.D: "move_right",
            key.S: "move_backward",
            key.W: "move_forward",
            key.Q: "move_down",
            key.E: "move_up",
        }

        # Load a TrueTypeFont plugin and open the font file
        self.display_font = text.FontManager().load_and_instantiate("TrueTypeFont")
        relative_path_to_font = "./ui_data/fonts/ProggyClean.ttf"
        self.display_font.open_file(
            os.path.join(os.path.dirname(__file__), relative_path_to_font),
            13,
        )

        # Glyphs we need to render everything
        self.glyph_cache = text.GlyphCache(mn.Vector2i(256))
        self.display_font.fill_glyph_cache(
            self.glyph_cache,
            string.ascii_lowercase
            + string.ascii_uppercase
            + string.digits
            + ":-_+,.! %µ",
        )

        # magnum text object that displays CPU/GPU usage data in the app window
        self.window_text = text.Renderer2D(
            self.display_font,
            self.glyph_cache,
            HabitatSimInteractiveViewer.DISPLAY_FONT_SIZE,
            text.Alignment.TOP_LEFT,
        )
        self.window_text.reserve(HabitatSimInteractiveViewer.MAX_DISPLAY_TEXT_CHARS)

        # text object transform in window space is Projection matrix times Translation Matrix
        # put text in top left of window
        self.window_text_transform = mn.Matrix3.projection(
            mn.Vector2(self.viewport_size)
        ) @ mn.Matrix3.translation(
            mn.Vector2(
                self.viewport_size[0]
                * -HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
                self.viewport_size[1]
                * HabitatSimInteractiveViewer.TEXT_DELTA_FROM_CENTER,
            )
        )
        self.shader = shaders.VectorGL2D()

        # make magnum text background transparent
        mn.gl.Renderer.enable(mn.gl.Renderer.Feature.BLENDING)
        mn.gl.Renderer.set_blend_function(
            mn.gl.Renderer.BlendFunction.ONE,
            mn.gl.Renderer.BlendFunction.ONE_MINUS_SOURCE_ALPHA,
        )
        mn.gl.Renderer.set_blend_equation(
            mn.gl.Renderer.BlendEquation.ADD, mn.gl.Renderer.BlendEquation.ADD
        )

        # variables that track app data and CPU/GPU usage
        self.num_frames_to_track = 60

        # Cycle mouse utilities
        self.mouse_interaction = MouseMode.LOOK
        self.mouse_grabber: Optional[MouseGrabber] = None
        self.previous_mouse_point = None

        # toggle physics simulation on/off
        self.simulating = True

        # toggle a single simulation step at the next opportunity if not
        # simulating continuously.
        self.simulate_single_step = False
        
        # configure our simulator
        self.cfg: Optional[habitat_sim.simulator.Configuration] = None
        self.sim: Optional[habitat_sim.simulator.Simulator] = None
        self.reconfigure_sim()

        # compute NavMesh if not already loaded by the scene.
        if (
            not self.sim.pathfinder.is_loaded
            and self.cfg.sim_cfg.scene_id.lower() != "none"
        ):
            self.navmesh_config_and_recompute()

        self.time_since_last_simulation = 0.0
        LoggingContext.reinitialize_from_env()
        logger.setLevel("INFO")
        self.print_help_text()

    def draw_contact_debug(self):
        """
        This method is called to render a debug line overlay displaying active contact points and normals.
        Yellow lines show the contact distance along the normal and red lines show the contact normal at a fixed length.
        """
        yellow = mn.Color4.yellow()
        red = mn.Color4.red()
        cps = self.sim.get_physics_contact_points()
        self.sim.get_debug_line_render().set_line_width(1.5)
        camera_position = self.render_camera.render_camera.node.absolute_translation
        # only showing active contacts
        active_contacts = (x for x in cps if x.is_active)
        for cp in active_contacts:
            # red shows the contact distance
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                cp.position_on_b_in_ws
                + cp.contact_normal_on_b_in_ws * -cp.contact_distance,
                red,
            )
            # yellow shows the contact normal at a fixed length for visualization
            self.sim.get_debug_line_render().draw_transformed_line(
                cp.position_on_b_in_ws,
                # + cp.contact_normal_on_b_in_ws * cp.contact_distance,
                cp.position_on_b_in_ws + cp.contact_normal_on_b_in_ws * 0.1,
                yellow,
            )
            self.sim.get_debug_line_render().draw_circle(
                translation=cp.position_on_b_in_ws,
                radius=0.005,
                color=yellow,
                normal=camera_position - cp.position_on_b_in_ws,
            )

    def debug_draw(self):
        """
        Additional draw commands to be called during draw_event.
        """
        if self.debug_bullet_draw:
            render_cam = self.render_camera.render_camera
            proj_mat = render_cam.projection_matrix.__matmul__(render_cam.camera_matrix)
            self.sim.physics_debug_draw(proj_mat)
        if self.contact_debug_draw:
            self.draw_contact_debug()

    def draw_event(
        self,
        simulation_call: Optional[Callable] = None,
        global_call: Optional[Callable] = None,
        active_agent_id_and_sensor_name: Tuple[int, str] = (0, "color_sensor"),
    ) -> None:
        """
        Calls continuously to re-render frames and swap the two frame buffers
        at a fixed rate.
        """
        agent_acts_per_sec = self.fps

        mn.gl.default_framebuffer.clear(
            mn.gl.FramebufferClear.COLOR | mn.gl.FramebufferClear.DEPTH
        )

        # Agent actions should occur at a fixed rate per second
        self.time_since_last_simulation += Timer.prev_frame_duration
        num_agent_actions: int = self.time_since_last_simulation * agent_acts_per_sec
        self.move_and_look(int(num_agent_actions))

        # Occasionally a frame will pass quicker than 1/60 seconds
        if self.time_since_last_simulation >= 1.0 / self.fps:
            if self.simulating or self.simulate_single_step:
                # step physics at a fixed rate
                # In the interest of frame rate, only a single step is taken,
                # even if time_since_last_simulation is quite large
                self.sim.step_world(1.0 / self.fps)
                self.simulate_single_step = False
                if simulation_call is not None:
                    simulation_call()
            if global_call is not None:
                global_call()

            # reset time_since_last_simulation, accounting for potential overflow
            self.time_since_last_simulation = math.fmod(
                self.time_since_last_simulation, 1.0 / self.fps
            )

        keys = active_agent_id_and_sensor_name

        self.sim._Simulator__sensors[keys[0]][keys[1]].draw_observation()
        # self.sim._Simulator__sensors[keys[0]]["depth_sensor"].draw_observation()
        agent = self.sim.get_agent(keys[0])
        self.render_camera = agent.scene_node.node_sensor_suite.get(keys[1])
        # self.render_camera = agent.scene_node.node_sensor_suite.get("depth_sensor")
        # self.debug_draw()
        self.render_camera.render_target.blit_rgba_to_default()
        mn.gl.default_framebuffer.bind()

        # draw CPU/GPU usage data and other info to the app window
        self.draw_text(self.render_camera.specification())

        self.swap_buffers()
        Timer.next_frame()
        self.redraw()

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        MOVE, LOOK = 0.07, 1.5

        # all of our possible actions' names
        action_list = [
            "move_left",
            "turn_left",
            "move_right",
            "turn_right",
            "move_backward",
            "look_up",
            "move_forward",
            "look_down",
            "move_down",
            "move_up",
        ]

        action_space: Dict[str, habitat_sim.agent.ActionSpec] = {}

        # build our action space map
        for action in action_list:
            actuation_spec_amt = MOVE if "move" in action else LOOK
            action_spec = make_action_spec(
                action, make_actuation_spec(actuation_spec_amt)
            )
            action_space[action] = action_spec

        sensor_spec: List[habitat_sim.sensor.SensorSpec] = self.cfg.agents[
            self.agent_id
        ].sensor_specifications

        agent_config = habitat_sim.agent.AgentConfiguration(
            height=1.5,
            radius=0.1,
            sensor_specifications=sensor_spec,
            action_space=action_space,
            body_type="cylinder",
        )
        return agent_config

    def reconfigure_sim(self) -> None:
        """
        Utilizes the current `self.sim_settings` to configure and set up a new
        `habitat_sim.Simulator`, and then either starts a simulation instance, or replaces
        the current simulator instance, reloading the most recently loaded scene
        """
        # configure our sim_settings but then set the agent to our default
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        self.sim = habitat_sim.Simulator(self.cfg)

        # post reconfigure
        self.active_scene_graph = self.sim.get_active_scene_graph()
        self.default_agent = self.sim.get_agent(self.agent_id)
        self.agent_body_node = self.default_agent.scene_node
        self.render_camera = self.agent_body_node.node_sensor_suite.get("color_sensor")
        # set sim_settings scene name as actual loaded scene
        self.sim_settings["scene"] = self.sim.curr_scene_name

        Timer.start()
        self.step = -1
        scene = self.sim.semantic_scene
        ins_id2label_id = {}
        label_id2name = {}
        neg_label_id = set()
        for obj in scene.objects:
            if obj is not None and obj.category is not None:
                label_id2name[obj.category.name()] = obj.category.index()
                if obj.category.index() < 0:
                    neg_label_id.add(obj.category.index())
                    print("[INFO] negative object index", obj.category.name(), obj.category.index())

                ins_id2label_id[int(obj.id.split("_")[-1])]= obj.category.index()
        

        label_ids = sorted(list(label_id2name.values()))
        ins_ids = sorted(list(ins_id2label_id.keys()))
        
        # TODO: append ignoring labelid to neg_label_id

        ins_id2ins_i = {}
        ins_id2label_pos = {}
        for ins_id in ins_ids:
            ins_i = ins_ids.index(ins_id)
            ins_id2ins_i[ins_id] = ins_i

            label_id = ins_id2label_id[ins_id]
            if label_id in neg_label_id:
                ins_id2label_pos[ins_id] = -100
            else:
                ins_id2label_pos[ins_id] = label_id+1 # color map append one background at beginning, so shift all label +1
        self.ins_id2ins_i = ins_id2ins_i
        self.ins_id2label_pos = ins_id2label_pos


    def move_and_look(self, repetitions: int) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        # avoids unecessary updates to grabber's object position
        if repetitions == 0:
            return

        key = Application.KeyEvent.Key
        agent = self.sim.agents[self.agent_id]
        press: Dict[key.key, bool] = self.pressed
        act: Dict[key.key, str] = self.key_to_action

        action_queue: List[str] = [act[k] for k, v in press.items() if v]
        for _ in range(int(repetitions)):
            for x in action_queue:
                agent.act(x)
                # observations = self.sim.step(x)
                # [agent.act(x) for x in action_queue]
                if self.total_frames % RECORD_FREQ ==0:
                    observations = self.sim.get_sensor_observations()
                    self.save_color_observation(observations, self.obs_save_idx)
                    self.save_semantic_observation(observations, self.obs_save_idx)
                    self.save_depth_observation(observations, self.obs_save_idx)
                    # self.save_semantic_observation_instance(observations, self.obs_save_idx)
                    self.save_traj()
                    self.obs_save_idx+=1
                self.total_frames += 1
        
        # update the grabber transform when our agent is moved
        # if self.mouse_grabber is not None:
        #     # update location of grabbed object
        #     self.update_grab_position(self.previous_mouse_point)

    # def invert_gravity(self) -> None:
    #     """
    #     Sets the gravity vector to the negative of it's previous value. This is
    #     a good method for testing simulation functionality.
    #     """
    #     gravity: mn.Vector3 = self.sim.get_gravity() * -1
    #     self.sim.set_gravity(gravity)

    def key_press_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key press by performing the corresponding functions.
        If the key pressed is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the
        key will be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key
        pressed = Application.KeyEvent.Key
        mod = Application.InputEvent.Modifier

        shift_pressed = bool(event.modifiers & mod.SHIFT)
        alt_pressed = bool(event.modifiers & mod.ALT)
        # warning: ctrl doesn't always pass through with other key-presses

        if key == pressed.ESC:
            event.accepted = True
            self.exit_event(Application.ExitEvent)
            return

        # elif key == pressed.H:
        #     self.print_help_text()

        # elif key == pressed.TAB:
        #     # NOTE: (+ALT) - reconfigure without cycling scenes
        #     if not alt_pressed:
        #         # cycle the active scene from the set available in MetadataMediator
        #         inc = -1 if shift_pressed else 1
        #         scene_ids = self.sim.metadata_mediator.get_scene_handles()
        #         cur_scene_index = 0
        #         if self.sim_settings["scene"] not in scene_ids:
        #             matching_scenes = [
        #                 (ix, x)
        #                 for ix, x in enumerate(scene_ids)
        #                 if self.sim_settings["scene"] in x
        #             ]
        #             if not matching_scenes:
        #                 logger.warning(
        #                     f"The current scene, '{self.sim_settings['scene']}', is not in the list, starting cycle at index 0."
        #                 )
        #             else:
        #                 cur_scene_index = matching_scenes[0][0]
        #         else:
        #             cur_scene_index = scene_ids.index(self.sim_settings["scene"])

        #         next_scene_index = min(
        #             max(cur_scene_index + inc, 0), len(scene_ids) - 1
        #         )
        #         self.sim_settings["scene"] = scene_ids[next_scene_index]
        #     self.reconfigure_sim()
        #     logger.info(
        #         f"Reconfigured simulator for scene: {self.sim_settings['scene']}"
        #     )

        # elif key == pressed.SPACE:
        #     if not self.sim.config.sim_cfg.enable_physics:
        #         logger.warn("Warning: physics was not enabled during setup")
        #     else:
        #         self.simulating = not self.simulating
        #         logger.info(f"Command: physics simulating set to {self.simulating}")

        # elif key == pressed.PERIOD:
        #     if self.simulating:
        #         logger.warn("Warning: physic simulation already running")
        #     else:
        #         self.simulate_single_step = True
        #         logger.info("Command: physics step taken")

        # elif key == pressed.COMMA:
        #     self.debug_bullet_draw = not self.debug_bullet_draw
        #     logger.info(f"Command: toggle Bullet debug draw: {self.debug_bullet_draw}")

        # elif key == pressed.C:
        #     if shift_pressed:
        #         self.contact_debug_draw = not self.contact_debug_draw
        #         logger.info(
        #             f"Command: toggle contact debug draw: {self.contact_debug_draw}"
        #         )
        #     else:
        #         # perform a discrete collision detection pass and enable contact debug drawing to visualize the results
        #         logger.info(
        #             "Command: perform discrete collision detection and visualize active contacts."
        #         )
        #         self.sim.perform_discrete_collision_detection()
        #         self.contact_debug_draw = True
        #         # TODO: add a nice log message with concise contact pair naming.

        # elif key == pressed.T:
        #     # load URDF
        #     fixed_base = alt_pressed
        #     urdf_file_path = ""
        #     if shift_pressed and self.cached_urdf:
        #         urdf_file_path = self.cached_urdf
        #     else:
        #         urdf_file_path = input("Load URDF: provide a URDF filepath:").strip()

        #     if not urdf_file_path:
        #         logger.warn("Load URDF: no input provided. Aborting.")
        #     elif not urdf_file_path.endswith((".URDF", ".urdf")):
        #         logger.warn("Load URDF: input is not a URDF. Aborting.")
        #     elif os.path.exists(urdf_file_path):
        #         self.cached_urdf = urdf_file_path
        #         aom = self.sim.get_articulated_object_manager()
        #         ao = aom.add_articulated_object_from_urdf(
        #             urdf_file_path, fixed_base, 1.0, 1.0, True
        #         )
        #         ao.translation = self.agent_body_node.transformation.transform_point(
        #             [0.0, 1.0, -1.5]
        #         )
        #     else:
        #         logger.warn("Load URDF: input file not found. Aborting.")

        # elif key == pressed.M:
        #     self.cycle_mouse_mode()
        #     logger.info(f"Command: mouse mode set to {self.mouse_interaction}")

        # elif key == pressed.V:
        #     self.invert_gravity()
        #     logger.info("Command: gravity inverted")

        elif key == pressed.N:
            # (default) - toggle navmesh visualization
            # NOTE: (+ALT) - re-sample the agent position on the NavMesh
            # NOTE: (+SHIFT) - re-compute the NavMesh
            if alt_pressed:
                logger.info("Command: resample agent state from navmesh")
                if self.sim.pathfinder.is_loaded:
                    new_agent_state = habitat_sim.AgentState()
                    new_agent_state.position = (
                        self.sim.pathfinder.get_random_navigable_point()
                    )
                    new_agent_state.rotation = quat_from_angle_axis(
                        self.sim.random.uniform_float(0, 2.0 * np.pi),
                        np.array([0, 1, 0]),
                    )
                    self.default_agent.set_state(new_agent_state)
                else:
                    logger.warning(
                        "NavMesh is not initialized. Cannot sample new agent state."
                    )
            elif shift_pressed:
                logger.info("Command: recompute navmesh")
                self.navmesh_config_and_recompute()
            else:
                if self.sim.pathfinder.is_loaded:
                    self.sim.navmesh_visualization = not self.sim.navmesh_visualization
                    logger.info("Command: toggle navmesh")
                else:
                    logger.warn("Warning: recompute navmesh first")

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = True
        event.accepted = True
        self.redraw()

    def map_by_dict(self, arr, mapping_dict):
        # NOTE: check missing meta
        missing_key = set(np.unique(arr))-mapping_dict.keys()
        for k in missing_key:
            mapping_dict[k] = -100

        return np.vectorize(mapping_dict.get)(arr)

    def save_color_observation(self, obs, total_frames):
        os.makedirs('test_output', exist_ok=True)
        os.makedirs('test_output/color', exist_ok=True)
        color_obs = obs["color_sensor"]
        color_img = Image.fromarray(color_obs, mode="RGBA")
        color_img.save("test_output/color/%d.png" % total_frames)
    
    def save_semantic_observation(self, obs, total_frames):
        os.makedirs('test_output', exist_ok=True)
        os.makedirs('test_output/sem', exist_ok=True)

        semantic_obs = obs["semantic_sensor"]
        label_obs = self.map_by_dict(semantic_obs, self.ins_id2label_pos)
        ins_i_obs = self.map_by_dict(semantic_obs, self.ins_id2ins_i) 

        np.save('test_output/sem/%d-label.npy'% total_frames, label_obs)
        np.save('test_output/sem/%d-instance.npy'% total_frames, ins_i_obs)

        label_obs = label_obs.flatten() 
        msk = label_obs==-100
        label_obs = label_obs % 40 + 1
        label_obs[msk] = 0
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((label_obs).astype(np.uint8))
        semantic_img.save("test_output/sem/%d-label.png" % total_frames)
        
        ins_i_obs = ins_i_obs.flatten() 
        msk = ins_i_obs==-100
        ins_i_obs = ins_i_obs % 40 + 1
        ins_i_obs[msk] = 0
        semantic_img_ins = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img_ins.putpalette(d3_40_colors_rgb.flatten())
        semantic_img_ins.putdata((ins_i_obs).astype(np.uint8))
        semantic_img_ins.save("test_output/sem/%d-ins.png" % total_frames)

    def save_depth_observation(self, obs, total_frames):
        os.makedirs('test_output', exist_ok=True)
        os.makedirs('test_output/depth', exist_ok=True)
        depth_obs = obs["depth_sensor"]
        # depth = (depth_obs * 1000).astype(int) # depth == 0 also means inf

        # f_path = "test_output/depth/%d.npy"%total_frames
        # np.save(f_path, depth)

        # depth_img = np.expand_dims(depth, -1).repeat(3, -1).astype(float)/np.amax(depth) * 255
        # depth_img = Image.fromarray(depth_img.astype(np.uint8), mode="RGB")
        # f_path_img = "test_output/depth/%d.png"%total_frames
        # depth_img.save(f_path_img)

        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        f_path_img = "test_output/depth/%d.png"%total_frames
        depth_img.save(f_path_img)

    def to_opengl_transform(self, transform=None):
        if transform is None:
            transform = np.eye(4)
        T = np.array([[1, 0, 0, 0],
                    [0, np.cos(np.pi), -np.sin(np.pi), 0],
                    [0, np.sin(np.pi), np.cos(np.pi), 0],
                    [0, 0, 0, 1]])
        return transform @ T

    def save_traj(self):
        state = self.sim.get_agent(0).get_state()
        sensor_state = state.sensor_states['color_sensor']
        print(sensor_state.rotation, state.rotation)

        cam_pose = np.eye(4)
        cam_pose[:3, 3] = sensor_state.position
        R = quat_to_magnum(sensor_state.rotation).to_matrix()
        cam_pose[:3, :3] = R
        cam_pose = self.to_opengl_transform(cam_pose)
        print(cam_pose[:3,3])
        # cam_pose = np.linalg.inv(cam_pose)

        with open('test_output/traj.txt', 'a') as f:
            for e in cam_pose.flatten():
                f.write(f"{e:.6f} ")
            f.write("\n")
            # f.write(" ".join([str(i) for i in cam_pose.flatten().tolist()]) + '\n')

        with jsonlines.open('test_output/traj.jsonl', mode='a') as writer:
            sim_pos = sensor_state.position
            sim_rot = quaternion.as_float_array(sensor_state.rotation)
            writer.write({
                "habitat_cam_pos": {"position": sim_pos.tolist(), "rotation":sim_rot.tolist()},
                "opengl_cam_pose": cam_pose.tolist()
            })

    def key_release_event(self, event: Application.KeyEvent) -> None:
        """
        Handles `Application.KeyEvent` on a key release. When a key is released, if it
        is part of the movement keys map `Dict[KeyEvent.key, Bool]`, then the key will
        be set to False for the next `self.move_and_look()` to update the current actions.
        """
        key = event.key

        # update map of moving/looking keys which are currently pressed
        if key in self.pressed:
            self.pressed[key] = False
        event.accepted = True
        self.redraw()


    # def mouse_move_event(self, event: Application.MouseMoveEvent) -> None:
    #     """
    #     Handles `Application.MouseMoveEvent`. When in LOOK mode, enables the left
    #     mouse button to steer the agent's facing direction. When in GRAB mode,
    #     continues to update the grabber's object positiion with our agents position.
    #     """
    #     button = Application.MouseMoveEvent.Buttons
    #     # if interactive mode -> LOOK MODE
    #     if event.buttons == button.LEFT and self.mouse_interaction == MouseMode.LOOK:
    #         agent = self.sim.agents[self.agent_id]
    #         delta = self.get_mouse_position(event.relative_position) / 2
    #         action = habitat_sim.agent.ObjectControls()
    #         act_spec = habitat_sim.agent.ActuationSpec

    #         # left/right on agent scene node
    #         action(agent.scene_node, "turn_right", act_spec(delta.x))

    #         # up/down on cameras' scene nodes
    #         action = habitat_sim.agent.ObjectControls()
    #         sensors = list(self.agent_body_node.subtree_sensors.values())
    #         [action(s.object, "look_down", act_spec(delta.y), False) for s in sensors]

    #     # if interactive mode is TRUE -> GRAB MODE
    #     elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
    #         # update location of grabbed object
    #         self.update_grab_position(self.get_mouse_position(event.position))

    #     self.previous_mouse_point = self.get_mouse_position(event.position)
    #     self.redraw()
    #     event.accepted = True

    # def mouse_press_event(self, event: Application.MouseEvent) -> None:
    #     """
    #     Handles `Application.MouseEvent`. When in GRAB mode, click on
    #     objects to drag their position. (right-click for fixed constraints)
    #     """
    #     button = Application.MouseEvent.Button
    #     physics_enabled = self.sim.get_physics_simulation_library()

    #     # if interactive mode is True -> GRAB MODE
    #     if self.mouse_interaction == MouseMode.GRAB and physics_enabled:
    #         render_camera = self.render_camera.render_camera
    #         ray = render_camera.unproject(self.get_mouse_position(event.position))
    #         raycast_results = self.sim.cast_ray(ray=ray)

    #         if raycast_results.has_hits():
    #             hit_object, ao_link = -1, -1
    #             hit_info = raycast_results.hits[0]

    #             if hit_info.object_id >= 0:
    #                 # we hit an non-staged collision object
    #                 ro_mngr = self.sim.get_rigid_object_manager()
    #                 ao_mngr = self.sim.get_articulated_object_manager()
    #                 ao = ao_mngr.get_object_by_id(hit_info.object_id)
    #                 ro = ro_mngr.get_object_by_id(hit_info.object_id)

    #                 if ro:
    #                     # if grabbed an object
    #                     hit_object = hit_info.object_id
    #                     object_pivot = ro.transformation.inverted().transform_point(
    #                         hit_info.point
    #                     )
    #                     object_frame = ro.rotation.inverted()
    #                 elif ao:
    #                     # if grabbed the base link
    #                     hit_object = hit_info.object_id
    #                     object_pivot = ao.transformation.inverted().transform_point(
    #                         hit_info.point
    #                     )
    #                     object_frame = ao.rotation.inverted()
    #                 else:
    #                     for ao_handle in ao_mngr.get_objects_by_handle_substring():
    #                         ao = ao_mngr.get_object_by_handle(ao_handle)
    #                         link_to_obj_ids = ao.link_object_ids

    #                         if hit_info.object_id in link_to_obj_ids:
    #                             # if we got a link
    #                             ao_link = link_to_obj_ids[hit_info.object_id]
    #                             object_pivot = (
    #                                 ao.get_link_scene_node(ao_link)
    #                                 .transformation.inverted()
    #                                 .transform_point(hit_info.point)
    #                             )
    #                             object_frame = ao.get_link_scene_node(
    #                                 ao_link
    #                             ).rotation.inverted()
    #                             hit_object = ao.object_id
    #                             break
    #                 # done checking for AO

    #                 if hit_object >= 0:
    #                     node = self.agent_body_node
    #                     constraint_settings = physics.RigidConstraintSettings()

    #                     constraint_settings.object_id_a = hit_object
    #                     constraint_settings.link_id_a = ao_link
    #                     constraint_settings.pivot_a = object_pivot
    #                     constraint_settings.frame_a = (
    #                         object_frame.to_matrix() @ node.rotation.to_matrix()
    #                     )
    #                     constraint_settings.frame_b = node.rotation.to_matrix()
    #                     constraint_settings.pivot_b = hit_info.point

    #                     # by default use a point 2 point constraint
    #                     if event.button == button.RIGHT:
    #                         constraint_settings.constraint_type = (
    #                             physics.RigidConstraintType.Fixed
    #                         )

    #                     grip_depth = (
    #                         hit_info.point - render_camera.node.absolute_translation
    #                     ).length()

    #                     self.mouse_grabber = MouseGrabber(
    #                         constraint_settings,
    #                         grip_depth,
    #                         self.sim,
    #                     )
    #                 else:
    #                     logger.warn("Oops, couldn't find the hit object. That's odd.")
    #             # end if didn't hit the scene
    #         # end has raycast hit
    #     # end has physics enabled

    #     self.previous_mouse_point = self.get_mouse_position(event.position)
    #     self.redraw()
    #     event.accepted = True

    # def mouse_scroll_event(self, event: Application.MouseScrollEvent) -> None:
    #     """
    #     Handles `Application.MouseScrollEvent`. When in LOOK mode, enables camera
    #     zooming (fine-grained zoom using shift) When in GRAB mode, adjusts the depth
    #     of the grabber's object. (larger depth change rate using shift)
    #     """
    #     scroll_mod_val = (
    #         event.offset.y
    #         if abs(event.offset.y) > abs(event.offset.x)
    #         else event.offset.x
    #     )
    #     if not scroll_mod_val:
    #         return

    #     # use shift to scale action response
    #     shift_pressed = bool(event.modifiers & Application.InputEvent.Modifier.SHIFT)
    #     alt_pressed = bool(event.modifiers & Application.InputEvent.Modifier.ALT)
    #     ctrl_pressed = bool(event.modifiers & Application.InputEvent.Modifier.CTRL)

    #     # if interactive mode is False -> LOOK MODE
    #     if self.mouse_interaction == MouseMode.LOOK:
    #         # use shift for fine-grained zooming
    #         mod_val = 1.01 if shift_pressed else 1.1
    #         mod = mod_val if scroll_mod_val > 0 else 1.0 / mod_val
    #         cam = self.render_camera
    #         cam.zoom(mod)
    #         self.redraw()

    #     elif self.mouse_interaction == MouseMode.GRAB and self.mouse_grabber:
    #         # adjust the depth
    #         mod_val = 0.1 if shift_pressed else 0.01
    #         scroll_delta = scroll_mod_val * mod_val
    #         if alt_pressed or ctrl_pressed:
    #             # rotate the object's local constraint frame
    #             agent_t = self.agent_body_node.transformation_matrix()
    #             # ALT - yaw
    #             rotation_axis = agent_t.transform_vector(mn.Vector3(0, 1, 0))
    #             if alt_pressed and ctrl_pressed:
    #                 # ALT+CTRL - roll
    #                 rotation_axis = agent_t.transform_vector(mn.Vector3(0, 0, -1))
    #             elif ctrl_pressed:
    #                 # CTRL - pitch
    #                 rotation_axis = agent_t.transform_vector(mn.Vector3(1, 0, 0))
    #             self.mouse_grabber.rotate_local_frame_by_global_angle_axis(
    #                 rotation_axis, mn.Rad(scroll_delta)
    #             )
    #         else:
    #             # update location of grabbed object
    #             self.mouse_grabber.grip_depth += scroll_delta
    #             self.update_grab_position(self.get_mouse_position(event.position))
    #     self.redraw()
    #     event.accepted = True

    def mouse_release_event(self, event: Application.MouseEvent) -> None:
        """
        Release any existing constraints.
        """
        del self.mouse_grabber
        self.mouse_grabber = None
        event.accepted = True

    def update_grab_position(self, point: mn.Vector2i) -> None:
        """
        Accepts a point derived from a mouse click event and updates the
        transform of the mouse grabber.
        """
        # check mouse grabber
        if not self.mouse_grabber:
            return

        render_camera = self.render_camera.render_camera
        ray = render_camera.unproject(point)

        rotation: mn.Matrix3x3 = self.agent_body_node.rotation.to_matrix()
        translation: mn.Vector3 = (
            render_camera.node.absolute_translation
            + ray.direction * self.mouse_grabber.grip_depth
        )
        self.mouse_grabber.update_transform(mn.Matrix4.from_(rotation, translation))

    def get_mouse_position(self, mouse_event_position: mn.Vector2i) -> mn.Vector2i:
        """
        This function will get a screen-space mouse position appropriately
        scaled based on framebuffer size and window size.  Generally these would be
        the same value, but on certain HiDPI displays (Retina displays) they may be
        different.
        """
        scaling = mn.Vector2i(self.framebuffer_size) / mn.Vector2i(self.window_size)
        return mouse_event_position * scaling

    def cycle_mouse_mode(self) -> None:
        """
        This method defines how to cycle through the mouse mode.
        """
        if self.mouse_interaction == MouseMode.LOOK:
            self.mouse_interaction = MouseMode.GRAB
        elif self.mouse_interaction == MouseMode.GRAB:
            self.mouse_interaction = MouseMode.LOOK

    def navmesh_config_and_recompute(self) -> None:
        """
        This method is setup to be overridden in for setting config accessibility
        in inherited classes.
        """
        self.navmesh_settings = habitat_sim.NavMeshSettings()
        self.navmesh_settings.set_defaults()
        self.navmesh_settings.agent_height = self.cfg.agents[self.agent_id].height
        self.navmesh_settings.agent_radius = self.cfg.agents[self.agent_id].radius

        self.sim.recompute_navmesh(
            self.sim.pathfinder,
            self.navmesh_settings,
            include_static_objects=True,
        )

    def exit_event(self, event: Application.ExitEvent):
        """
        Overrides exit_event to properly close the Simulator before exiting the
        application.
        """
        self.sim.close(destroy=True)
        event.accepted = True
        exit(0)

    def draw_text(self, sensor_spec):
        self.shader.bind_vector_texture(self.glyph_cache.texture)
        self.shader.transformation_projection_matrix = self.window_text_transform
        self.shader.color = [1.0, 1.0, 1.0]

        sensor_type_string = str(sensor_spec.sensor_type.name)
        sensor_subtype_string = str(sensor_spec.sensor_subtype.name)
        if self.mouse_interaction == MouseMode.LOOK:
            mouse_mode_string = "LOOK"
        elif self.mouse_interaction == MouseMode.GRAB:
            mouse_mode_string = "GRAB"
        self.window_text.render(
            f"""
                {self.fps} FPS
                Sensor Type: {sensor_type_string}
                Sensor Subtype: {sensor_subtype_string}
                Mouse Interaction Mode: {mouse_mode_string}
            """
        )
        self.shader.draw(self.window_text.mesh)

    def print_help_text(self) -> None:
        """
        Print the Key Command help text.
        """
        logger.info(
            """
=====================================================
Welcome to the Habitat-sim Python Viewer application!
=====================================================
Mouse Functions ('m' to toggle mode):
----------------
In LOOK mode (default):
    LEFT:
        Click and drag to rotate the agent and look up/down.
    WHEEL:
        Modify orthographic camera zoom/perspective camera FOV (+SHIFT for fine grained control)

In GRAB mode (with 'enable-physics'):
    LEFT:
        Click and drag to pickup and move an object with a point-to-point constraint (e.g. ball joint).
    RIGHT:
        Click and drag to pickup and move an object with a fixed frame constraint.
    WHEEL (with picked object):
        default - Pull gripped object closer or push it away.
        (+ALT) rotate object fixed constraint frame (yaw)
        (+CTRL) rotate object fixed constraint frame (pitch)
        (+ALT+CTRL) rotate object fixed constraint frame (roll)
        (+SHIFT) amplify scroll magnitude


Key Commands:
-------------
    esc:        Exit the application.
    'h':        Display this help message.
    'm':        Cycle mouse interaction modes.

    Agent Controls:
    'wasd':     Move the agent's body forward/backward and left/right.
    'zx':       Move the agent's body up/down.
    arrow keys: Turn the agent's body left/right and camera look up/down.

    Utilities:
    'r':        Reset the simulator with the most recently loaded scene.
    'n':        Show/hide NavMesh wireframe.
                (+SHIFT) Recompute NavMesh with default settings.
                (+ALT) Re-sample the agent(camera)'s position and orientation from the NavMesh.
    ',':        Render a Bullet collision shape debug wireframe overlay (white=active, green=sleeping, blue=wants sleeping, red=can't sleep).
    'c':        Run a discrete collision detection pass and render a debug wireframe overlay showing active contact points and normals (yellow=fixed length normals, red=collision distances).
                (+SHIFT) Toggle the contact point debug render overlay on/off.

    Object Interactions:
    SPACE:      Toggle physics simulation on/off.
    '.':        Take a single simulation step if not simulating continuously.
    'v':        (physics) Invert gravity.
    't':        Load URDF from filepath
                (+SHIFT) quick re-load the previously specified URDF
                (+ALT) load the URDF with fixed base
=====================================================
"""
        )


class MouseMode(Enum):
    LOOK = 0
    GRAB = 1
    MOTION = 2


class MouseGrabber:
    """
    Create a MouseGrabber from RigidConstraintSettings to manipulate objects.
    """

    def __init__(
        self,
        settings: physics.RigidConstraintSettings,
        grip_depth: float,
        sim: habitat_sim.simulator.Simulator,
    ) -> None:
        self.settings = settings
        self.simulator = sim

        # defines distance of the grip point from the camera for pivot updates
        self.grip_depth = grip_depth
        self.constraint_id = sim.create_rigid_constraint(settings)

    def __del__(self):
        self.remove_constraint()

    def remove_constraint(self) -> None:
        """
        Remove a rigid constraint by id.
        """
        self.simulator.remove_rigid_constraint(self.constraint_id)

    def updatePivot(self, pos: mn.Vector3) -> None:
        self.settings.pivot_b = pos
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_frame(self, frame: mn.Matrix3x3) -> None:
        self.settings.frame_b = frame
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def update_transform(self, transform: mn.Matrix4) -> None:
        self.settings.frame_b = transform.rotation()
        self.settings.pivot_b = transform.translation
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)

    def rotate_local_frame_by_global_angle_axis(
        self, axis: mn.Vector3, angle: mn.Rad
    ) -> None:
        """rotate the object's local constraint frame with a global angle axis input."""
        object_transform = mn.Matrix4()
        rom = self.simulator.get_rigid_object_manager()
        aom = self.simulator.get_articulated_object_manager()
        if rom.get_library_has_id(self.settings.object_id_a):
            object_transform = rom.get_object_by_id(
                self.settings.object_id_a
            ).transformation
        else:
            # must be an ao
            object_transform = (
                aom.get_object_by_id(self.settings.object_id_a)
                .get_link_scene_node(self.settings.link_id_a)
                .transformation
            )
        local_axis = object_transform.inverted().transform_vector(axis)
        R = mn.Matrix4.rotation(angle, local_axis.normalized())
        self.settings.frame_a = R.rotation().__matmul__(self.settings.frame_a)
        self.simulator.update_rigid_constraint(self.constraint_id, self.settings)


class Timer:
    """
    Timer class used to keep track of time between buffer swaps
    and guide the display frame rate.
    """

    start_time = 0.0
    prev_frame_time = 0.0
    prev_frame_duration = 0.0
    running = False

    @staticmethod
    def start() -> None:
        """
        Starts timer and resets previous frame time to the start time
        """
        Timer.running = True
        Timer.start_time = time.time()
        Timer.prev_frame_time = Timer.start_time
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def stop() -> None:
        """
        Stops timer and erases any previous time data, reseting the timer
        """
        Timer.running = False
        Timer.start_time = 0.0
        Timer.prev_frame_time = 0.0
        Timer.prev_frame_duration = 0.0

    @staticmethod
    def next_frame() -> None:
        """
        Records previous frame duration and updates the previous frame timestamp
        to the current time. If the timer is not currently running, perform nothing.
        """
        if not Timer.running:
            return
        Timer.prev_frame_duration = time.time() - Timer.prev_frame_time
        Timer.prev_frame_time = time.time()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    # optional arguments
    parser.add_argument(
        "--scene",
        default="./data/test_assets/scenes/simple_room.glb",
        type=str,
        help='scene/stage file to load (default: "./data/test_assets/scenes/simple_room.glb")',
    )
    parser.add_argument(
        "--dataset",
        default="./data/objects/ycb/ycb.scene_dataset_config.json",
        type=str,
        metavar="DATASET",
        help='dataset configuration file to use (default: "./data/objects/ycb/ycb.scene_dataset_config.json")',
    )

    args = parser.parse_args()

    # Setting up sim_settings
    sim_settings: Dict[str, Any] = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    
    # NOTE activate semantic sensor
    sim_settings['color_sensor'] = True
    sim_settings['semantic_sensor'] = True
    sim_settings['depth_sensor'] = True

    sim_settings["enable_physics"] = False

    sim_settings["seed"] = 199

    # start the application
    HabitatSimInteractiveViewer(sim_settings).exec()
