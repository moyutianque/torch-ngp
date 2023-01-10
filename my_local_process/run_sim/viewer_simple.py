import habitat_sim
from habitat_sim.utils.settings import default_sim_settings, make_cfg
import numpy as np
from PIL import Image
from habitat_sim.utils.common import quat_to_magnum
import os
from typing import Any, Callable, Dict, List, Optional, Tuple
import quaternion
import jsonlines
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
RECORD_FREQ  = 1

keys_dict = {
    'i': "look_up",
    'k': "look_down",
    'j': "turn_left",
    'l': "turn_right",
    'a': "move_left",
    'd': "move_right",
    's': "move_backward",
    'w': "move_forward",
    'q': 'exit'
}

class Viewer:
    def __init__(self, sim_settings):
        # configure our sim_settings but then set the agent to our default
        self.sim_settings = sim_settings
        self.cfg = make_cfg(self.sim_settings)
        self.agent_id: int = self.sim_settings["default_agent"]
        self.cfg.agents[self.agent_id] = self.default_agent_config()

        self.sim = habitat_sim.Simulator(self.cfg)
        
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

        self.obs_save_idx = 0
        self.total_frames = 0

    def default_agent_config(self) -> habitat_sim.agent.AgentConfiguration:
        """
        Set up our own agent and agent controls
        """
        make_action_spec = habitat_sim.agent.ActionSpec
        make_actuation_spec = habitat_sim.agent.ActuationSpec
        # MOVE, LOOK = 0.07, 1.5
        MOVE, LOOK = 0.5, 30

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
        return color_img
    
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
        depth = (depth_obs * 1000).astype(int) # depth == 0 also means inf

        f_path = "test_output/depth/%d.npy"%total_frames
        np.save(f_path, depth)

        depth_img = np.expand_dims(depth, -1).repeat(3, -1).astype(float)/np.amax(depth) * 255
        depth_img = Image.fromarray(depth_img.astype(np.uint8), mode="RGB")
        f_path_img = "test_output/depth/%d.png"%total_frames
        depth_img.save(f_path_img)

        # depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        # f_path_img = "test_output/depth/%d.png"%total_frames
        # depth_img.save(f_path_img)

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


    def move_and_look(self) -> None:
        """
        This method is called continuously with `self.draw_event` to monitor
        any changes in the movement keys map `Dict[KeyEvent.key, Bool]`.
        When a key in the map is set to `True` the corresponding action is taken.
        """
        # avoids unecessary updates to grabber's object position

        agent = self.sim.agents[self.agent_id]
        while True:
            key = input()
            if key not in keys_dict:
                continue
            x = keys_dict[key]
            if x == 'exit':
                break

            # agent.act(x)
            observations = self.sim.step(x)
            # [agent.act(x) for x in action_queue]
            if self.total_frames % RECORD_FREQ ==0:
                # observations = self.sim.get_sensor_observations()
                img = self.save_color_observation(observations, self.obs_save_idx)
                self.save_semantic_observation(observations, self.obs_save_idx)
                self.save_depth_observation(observations, self.obs_save_idx)
                # self.save_semantic_observation_instance(observations, self.obs_save_idx)
                self.save_traj()
                img.show()
                self.obs_save_idx+=1
            self.total_frames += 1

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
    sim_settings = default_sim_settings
    sim_settings["scene"] = args.scene
    sim_settings["scene_dataset_config_file"] = args.dataset
    
    # NOTE activate semantic sensor
    sim_settings['color_sensor'] = True
    sim_settings['semantic_sensor'] = True
    sim_settings['depth_sensor'] = True

    sim_settings["enable_physics"] = False

    sim_settings["seed"] = 199

    # start the application
    Viewer(sim_settings).move_and_look()