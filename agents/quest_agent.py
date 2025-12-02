import time
import mujoco
import numpy as np
from typing import Literal, Dict
from agents.agent import Agent
from utils.rotations import matrix_to_quat_wxyz
from utils.utils import MovingAverageQueue
from utils.sound import play_beep
from dm_control import mjcf
from utils.ik import qpos_from_site_pose
from oculus_reader.reader import OculusReader
import mujoco
import mujoco.viewer
# assuming looking from robot back
# right quest to right hand
# what we want:
# quest [1, 0, 0] -> gr1 [0, 1, 0] -> first column
# quest [0, 1, 0] -> gr1 [0, 0, 1] -> second column
# quest [0, 0, 1] -> gr1 [1, 0, 0] -> third column
quest2gr1 = np.array([
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
])

# gr1 [1, 0, 0] -> quest [0, 0, 1] -> first column
# gr1 [0, 1, 0] -> quest [1, 0, 0] -> second column
# gr1 [0, 0, 1] -> quest [0, 1, 0] -> third column
gr12quest = np.array([
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
])


class SingleArmQuestAgent(Agent):
    def __init__(
        self,
        robot_type: Literal["GR1", "STAR1"],
        which_hand: Literal["l", "r"],
    ) -> None:
        # quest controller
        self.oculus_reader = OculusReader()
        # init or when trigger button is released
        self.quest_last_xyz, self.quest_last_mat = None, None
        self.robot_last_xyz, self.robot_last_mat = None, None
        self.which_hand = which_hand
        self.site_name = 'right_hand' if self.which_hand == 'r' else 'left_hand'
        self.trigger_key = 'rightTrig' if self.which_hand == 'r' else 'leftTrig'
        self.grip_key = 'rightGrip' if self.which_hand == 'r' else 'leftGrip'
        self.joystick_key = 'rightJS' if self.which_hand == 'r' else 'leftJS'
        # when right joystick pressed, enter waist mode
        self.waist_activated_key = 'RJ'
        self.waist_activated = False
        self.ref_waist_q = [0.0, -0.1, 0.0]
        self.robot_type = robot_type
        if self.robot_type == 'GR1':
            mjcf_model = mjcf.from_path('assets/gr1t2.xml')
            self.model = mujoco.MjModel.from_xml_path("assets/gr1t2.xml")
            self.num_qpos = 32
            self.left_arm_idx = (18, 25)
            self.right_arm_idx = (25, 32)
            left_arm_init = [0.56685759, 0.27324125, 0.08566903, -1.74873004, 0.03957806, 0.00512999, -0.11432972]
            right_arm_init = [0.56685759, -0.27324125, 0.08566903, -1.74873004, 0.03957806, 0.00512999, -0.11432972]
        elif self.robot_type == "STAR1":
            mjcf_model = mjcf.from_path('assets/star1/l3_new.xml')
            self.model = mujoco.MjModel.from_xml_path("assets/star1/l3_new.xml")
            self.num_qpos = 62
            self.left_arm_idx = (9, 16)
            self.right_arm_idx = (28, 35)
            left_arm_init = [0, 0, 0, -0.5, 0, 0, 0]
            right_arm_init = [0, 0, 0, -0.5, 0, 0, 0]
        else:
            raise NotImplementedError

        self.physics = mjcf.Physics.from_mjcf_model(mjcf_model)
        self.physics.model.opt.gravity[:] = 0.0
        self.physics.data.qpos[self.left_arm_idx[0]:self.left_arm_idx[1]] = np.array([left_arm_init])
        self.physics.data.qpos[self.right_arm_idx[0]:self.right_arm_idx[1]] = np.array([right_arm_init])

        if self.which_hand == 'r':
            arm_actions = self.physics.data.qpos[self.right_arm_idx[0]:self.right_arm_idx[1]]
        else:
            arm_actions = self.physics.data.qpos[self.left_arm_idx[0]:self.left_arm_idx[1]]
        self.last_action = np.concatenate([arm_actions, np.zeros(6)])
        self.action_queue = MovingAverageQueue(20, self.last_action.shape[0], 0.7)
        for i in range(20):
            self.action_queue.add(self.last_action)
        self.reference_js = [0.5, 0.5]
        self.js_speed_scale = 0.02
        self.start_tracking_timestamp = 0

        self.ik_joints = self._filter_ik_joints(mjcf_model)
        self.hand_eye_data = mujoco.MjData(self.model)

    def act(self, obs: Dict[str, np.ndarray]) -> np.ndarray:
        pose_data, button_data = self.oculus_reader.get_transformations_and_buttons()
        assert obs['qpos'].shape == (self.num_qpos,)
        self.physics.data.qpos = obs['qpos']
        self.physics.step()
        if self.quest_last_xyz is None or self.quest_last_mat is None:
            self.quest_last_xyz = pose_data[self.which_hand][:, -1]
            self.quest_last_mat = pose_data[self.which_hand][:3, :3]
            self.robot_last_xyz = np.array(self.physics.named.data.site_xpos[self.site_name])
            self.robot_last_mat = np.array(self.physics.named.data.site_xmat[self.site_name]).reshape(3, 3)

        # deal with the waist
        if self.which_hand == 'r' and button_data[self.waist_activated_key]:
            # when switching control mode, frozen a bit to avoid frequent update
            time.sleep(0.2)
            self.waist_activated = not self.waist_activated
        self._waist_mapping(button_data)

        stop_eye_tracking = False
        if self.start_tracking_timestamp > 0:
            if (self.which_hand == "r" and button_data["B"]) or (self.which_hand == "l" and button_data["Y"]):
                stop_eye_tracking = True

        # deal with the head
        if (self.which_hand == "r" and button_data["A"]) or (self.which_hand == "l" and button_data["X"]):
            self.start_tracking_timestamp = time.time()

        if self.start_tracking_timestamp > 0:
            head_actions = self._eye_hand_tracking()
        else:
            head_actions = np.zeros(3)

        # parse hand data
        if not self.waist_activated:
            if self.which_hand == "l":
                js_y = button_data[self.joystick_key][0]
                js_x = button_data[self.joystick_key][1] * -1
            else:
                js_y = button_data[self.joystick_key][0] * -1
                js_x = button_data[self.joystick_key][1] * -1
            self.reference_js = [
                max(0, min(self.reference_js[0] + js_x * self.js_speed_scale, 1)),
                max(0, min(self.reference_js[1] + js_y * self.js_speed_scale, 1)),
            ]

        hand_actions = [button_data[self.grip_key][0]] * 4 + self.reference_js
        # get relative transformation from (current) to (last deactivated) status
        triggered = button_data[self.trigger_key][0] > 0.5

        if triggered:
            delta_quest_xyz = pose_data[self.which_hand][:, -1] - self.quest_last_xyz
            delta_quest_mat = pose_data[self.which_hand][:3, :3] @ self.quest_last_mat.T
            # translation
            delta_gr1_xyz = (quest2gr1 @ delta_quest_xyz)[:3]
            gr1_xyz = self.robot_last_xyz + delta_gr1_xyz
            # rotation
            gr1_mat = np.eye(4)
            gr1_mat[:3, :3] = self.robot_last_mat
            quest_mat = gr12quest @ gr1_mat
            quest_mat[:3, :3] = delta_quest_mat @ quest_mat[:3, :3]
            gr1_mat = quest2gr1 @ quest_mat
            gr1_quat = matrix_to_quat_wxyz(gr1_mat[:3, :3])
            ik = qpos_from_site_pose(
                self.physics, self.site_name,
                target_pos=gr1_xyz,
                target_quat=gr1_quat,
                joint_names=self.ik_joints,
                max_steps=400,
                max_update_norm=1.0,
            )
            self.physics.reset()
            if ik.success:
                # TODO: this is a choice: if we only care about a single arm?
                # currently we only output 7-dimensional
                if self.which_hand == 'r':
                    arm_actions = ik.qpos[self.right_arm_idx[0]:self.right_arm_idx[1]]
                else:
                    arm_actions = ik.qpos[self.left_arm_idx[0]:self.left_arm_idx[1]]

                if self.which_hand == 'l':
                    arm_actions = np.clip(
                        arm_actions,
                        [-1.70, -0.30, -2.50, -2.0, -2.60, -0.40, -0.50],
                        [1.00, 1.85, 2.50, 1.15, 2.60, 0.50, 0.40]
                    )
                else:
                    arm_actions = np.clip(
                        arm_actions,
                        [-1.70, -1.80, -2.50, -2.0, -2.60, -0.40, -0.50],
                        [1.00, 0.30, 2.50, 1.15, 2.60, 0.50, 0.40]
                    )
                self.last_action = np.concatenate([arm_actions, hand_actions])
                ave_action = self.action_queue.add(self.last_action)
            else:
                self.last_action[-6:] = hand_actions
                # Example: Play a beep
                # play_beep(frequency=2000, duration=0.05)
                ave_action = self.action_queue.add(self.last_action)
        else:
            self.quest_last_xyz = pose_data[self.which_hand][:, -1]
            self.quest_last_mat = pose_data[self.which_hand][:3, :3]
            self.robot_last_xyz = np.array(self.physics.named.data.site_xpos[self.site_name])
            self.robot_last_mat = np.array(self.physics.named.data.site_xmat[self.site_name]).reshape(3, 3)
            self.last_action[-6:] = hand_actions
            ave_action = self.action_queue.add(self.last_action)

        return ave_action, head_actions, self.start_tracking_timestamp, self.ref_waist_q, stop_eye_tracking

    def _filter_ik_joints(self, mjcf_model):
        all_joint_names = [joint.name for joint in mjcf_model.find_all('joint')]
        ik_joints = []
        for j in all_joint_names:
            if j is None:
                continue
            hand = "left" if self.which_hand == "l" else "right"
            if self.robot_type == "GR1":
                if f'{hand}_shoulder' in j or f'{hand}_elbow' in j or f'{hand}_wrist' in j:
                    ik_joints.append(j)
            elif self.robot_type == "STAR1":
                if f'{hand}_shoulder' in j or f'{hand}_elbow' in j or f'{hand}_wrist' in j or f'{hand}_arm' in j:
                    ik_joints.append(j)
            else:
                raise NotImplementedError
        print(ik_joints)
        return ik_joints

    def _waist_mapping(self, button_data):
        if self.waist_activated:
            js_y = button_data[self.joystick_key][0]
            js_x = button_data[self.joystick_key][1]
            if abs(js_x) > 0.95 and abs(js_y) < 0.4:
                self.ref_waist_q[1] += np.sign(js_x) * 0.005
                print(self.ref_waist_q[1])
                self.ref_waist_q[1] = np.clip(self.ref_waist_q[1], a_min=-0.25, a_max=0.1)
            if abs(js_x) < 0.4 and abs(js_y) > 0.95:
                self.ref_waist_q[2] += np.sign(js_y) * 0.005
                self.ref_waist_q[2] = np.clip(self.ref_waist_q[2], -0.25, 0.25)
            if abs(js_x) > 0.4 and abs(js_y) > 0.4:
                self.ref_waist_q[0] += np.sign(js_y) * 0.005
                self.ref_waist_q[0] = np.clip(self.ref_waist_q[0], a_min=-0.25, a_max=0.25)

    def _eye_hand_tracking(self):
        model = self.model
        # Get position of head end-effector (after yaw joint) in world coordinates.
        self.hand_eye_data.qpos[:] = self.physics.data.qpos
        mujoco.mj_step(model, self.hand_eye_data)
        # self.mujoco_debug_viewer.sync()
        data = self.hand_eye_data
        head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'head_red_ball')
        head_pos = data.geom_xpos[head_id]

        # Get hand position and orientation based on side
        # Get hand position and orientation based on side.
        if self.which_hand == "r":
            hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'right_red_ball')
        else:
            hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'left_red_ball')
        hand_pos = data.geom_xpos[hand_id]
        hand_rot = data.geom_xmat[hand_id].reshape(3, 3)

        # Transform offset to world frame.
        palm_position_wrt_hand = np.array([0.0, 0.0, 0.0])
        world_offset = hand_rot @ np.asarray(palm_position_wrt_hand)
        palm_pos = hand_pos + world_offset

        # Calculate relative position vector in world frame.
        rel_pos = palm_pos - head_pos

        # Get joint IDs and their limits.
        pitch_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_pitch"
        )
        roll_joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_roll"
        )
        yaw_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint_head_yaw")

        # Usual Euler angles calculation.
        xy_dist = np.sqrt(rel_pos[0] ** 2 + rel_pos[1] ** 2)
        pitch = np.arctan2(
            -rel_pos[2], xy_dist
        )  # Negative because positive pitch looks down
        pitch = np.clip(
            pitch, model.jnt_range[pitch_joint_id][0], model.jnt_range[pitch_joint_id][1]
        )
        roll = 0.0
        roll = np.clip(
            roll, model.jnt_range[roll_joint_id][0], model.jnt_range[roll_joint_id][1]
        )
        yaw = np.arctan2(rel_pos[1], rel_pos[0])
        yaw = np.clip(
            yaw, model.jnt_range[yaw_joint_id][0], model.jnt_range[yaw_joint_id][1]
        )
        return np.array([pitch, roll, yaw])


class DualArmQuestAgent(Agent):
    def __init__(self, robot_type: Literal["GR1", "STAR1"],):
        self.agent_left = SingleArmQuestAgent(robot_type=robot_type, which_hand='l')
        self.agent_right = SingleArmQuestAgent(robot_type=robot_type, which_hand='r')
        self.head_tracking_stamp_left = 0
        self.head_tracking_stamp_right = 0
        self.head_action_queue = MovingAverageQueue(10, 3, 0.3)
        self.last_head_actions = np.array([0.6, 0, 0])
        self.stop_head = False
        for i in range(50):
            self.head_action_queue.add(np.zeros(3))

    def act(self, obs: Dict) -> np.ndarray:
        left_arm_hand_actions, head_actions_left, self.head_tracking_stamp_left, _, stop_head_left = self.agent_left.act(obs)
        right_arm_hand_actions, head_actions_right, self.head_tracking_stamp_right, waist_actions, stop_head_right = self.agent_right.act(obs)

        if stop_head_left and not self.stop_head:
            self.stop_head = True
        if stop_head_right and not self.stop_head:
            self.stop_head = True

        if not self.stop_head:
            if self.head_tracking_stamp_left == 0 and self.head_tracking_stamp_right == 0:
                head_actions = np.array([0.6, 0, 0])
            else:
                if self.head_tracking_stamp_left > self.head_tracking_stamp_right:
                    head_actions = head_actions_left
                    head_actions[2] += 0.05
                else:
                    head_actions = head_actions_right
                    head_actions[2] += 0.05
        else:
            head_actions = self.last_head_actions

        # enforce norm
        head_move_norm = np.linalg.norm(head_actions - self.last_head_actions)
        head_move_norm = min(head_move_norm, 0.05)
        head_actions = self.last_head_actions + (head_actions - self.last_head_actions) * head_move_norm
        head_actions = np.clip(head_actions, [-0.4, -1000, -10000], [0.4, 1000, 1000])
        self.last_head_actions = head_actions.copy()
        arm_hand_actions = np.concatenate([left_arm_hand_actions, right_arm_hand_actions])
        return arm_hand_actions, head_actions, waist_actions
