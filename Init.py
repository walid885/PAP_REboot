import mujoco
import mujoco.viewer
import numpy as np
import time
import threading

class ValkyriePickPlacePOC:
    def __init__(self):
        # Create Valkyrie XML model string
        self.model_xml = self._create_valkyrie_xml()
        
        # Load model and create data
        self.model = mujoco.MjModel.from_xml_string(self.model_xml)
        self.data = mujoco.MjData(self.model)
        
        # Initialize observation and action dimensions
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()
        
        # Target object position for pick and place
        self.target_object_pos = np.array([0.5, 0.0, 0.8])
        self.place_target_pos = np.array([0.3, 0.3, 0.8])
        
        # Episode tracking
        self.episode_step = 0
        self.max_episode_steps = 1000
        
    def _create_valkyrie_xml(self):
        """Create simplified Valkyrie humanoid XML model"""
        xml = '''
        <mujoco model="valkyrie_humanoid">
            <compiler angle="radian" meshdir="." texturedir="."/>
            
            <option timestep="0.002" iterations="50" solver="PGS" gravity="0 0 -9.81"/>
            
            <asset>
                <texture name="grid" type="2d" builtin="checker" rgb1="0.1 0.2 0.3" 
                         rgb2="0.2 0.3 0.4" width="300" height="300"/>
                <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.2"/>
            </asset>
            
            <worldbody>
                <light pos="0 0 3" dir="0 0 -1" diffuse="1 1 1"/>
                <geom name="floor" size="10 10 0.1" type="plane" material="grid"/>
                
                <!-- Target object to pick -->
                <body name="target_object" pos="0.5 0.0 0.8">
                    <geom name="target_cube" type="box" size="0.03 0.03 0.03" 
                          rgba="1 0 0 1" mass="0.1"/>
                    <joint name="target_free" type="free"/>
                </body>
                
                <!-- Place target marker -->
                <body name="place_target" pos="0.3 0.3 0.8">
                    <geom name="place_marker" type="sphere" size="0.02" 
                          rgba="0 1 0 0.5" mass="0.001"/>
                </body>
                
                <!-- Valkyrie Humanoid -->
                <body name="torso" pos="0 0 1.0">
                    <geom name="torso" type="capsule" size="0.07 0.3" rgba="0.8 0.8 0.8 1"/>
                    <joint name="root" type="free"/>
                    
                    <!-- Head -->
                    <body name="head" pos="0 0 0.35">
                        <geom name="head" type="sphere" size="0.08" rgba="0.9 0.9 0.9 1"/>
                        <joint name="neck" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                    </body>
                    
                    <!-- Right Arm -->
                    <body name="right_upper_arm" pos="0.15 0 0.2">
                        <geom name="r_upper_arm" type="capsule" size="0.03 0.15" rgba="0.7 0.7 0.7 1"/>
                        <joint name="r_shoulder_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <joint name="r_shoulder_roll" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        
                        <body name="right_lower_arm" pos="0 0 -0.15">
                            <geom name="r_lower_arm" type="capsule" size="0.025 0.12" rgba="0.6 0.6 0.6 1"/>
                            <joint name="r_elbow" type="hinge" axis="0 1 0" range="0 2.8"/>
                            
                            <body name="right_hand" pos="0 0 -0.12">
                                <geom name="r_hand" type="box" size="0.04 0.02 0.06" rgba="0.5 0.5 0.5 1"/>
                                <joint name="r_wrist" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
                                
                                <!-- Simple gripper -->
                                <body name="r_finger1" pos="0.03 0 0.03">
                                    <geom name="r_f1" type="box" size="0.01 0.01 0.02" rgba="0.3 0.3 0.3 1"/>
                                    <joint name="r_finger1" type="hinge" axis="0 1 0" range="-0.3 0.3"/>
                                </body>
                                <body name="r_finger2" pos="0.03 0 -0.03">
                                    <geom name="r_f2" type="box" size="0.01 0.01 0.02" rgba="0.3 0.3 0.3 1"/>
                                    <joint name="r_finger2" type="hinge" axis="0 1 0" range="-0.3 0.3"/>
                                </body>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Left Arm -->
                    <body name="left_upper_arm" pos="-0.15 0 0.2">
                        <geom name="l_upper_arm" type="capsule" size="0.03 0.15" rgba="0.7 0.7 0.7 1"/>
                        <joint name="l_shoulder_pitch" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
                        <joint name="l_shoulder_roll" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
                        
                        <body name="left_lower_arm" pos="0 0 -0.15">
                            <geom name="l_lower_arm" type="capsule" size="0.025 0.12" rgba="0.6 0.6 0.6 1"/>
                            <joint name="l_elbow" type="hinge" axis="0 1 0" range="0 2.8"/>
                            
                            <body name="left_hand" pos="0 0 -0.12">
                                <geom name="l_hand" type="box" size="0.04 0.02 0.06" rgba="0.5 0.5 0.5 1"/>
                                <joint name="l_wrist" type="hinge" axis="0 1 0" range="-1.0 1.0"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Right Leg -->
                    <body name="right_thigh" pos="0.08 0 -0.35">
                        <geom name="r_thigh" type="capsule" size="0.04 0.18" rgba="0.7 0.7 0.7 1"/>
                        <joint name="r_hip_pitch" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                        <joint name="r_hip_roll" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
                        
                        <body name="right_shin" pos="0 0 -0.18">
                            <geom name="r_shin" type="capsule" size="0.035 0.15" rgba="0.6 0.6 0.6 1"/>
                            <joint name="r_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                            
                            <body name="right_foot" pos="0 0 -0.15">
                                <geom name="r_foot" type="box" size="0.08 0.04 0.02" rgba="0.5 0.5 0.5 1"/>
                                <joint name="r_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                            </body>
                        </body>
                    </body>
                    
                    <!-- Left Leg -->
                    <body name="left_thigh" pos="-0.08 0 -0.35">
                        <geom name="l_thigh" type="capsule" size="0.04 0.18" rgba="0.7 0.7 0.7 1"/>
                        <joint name="l_hip_pitch" type="hinge" axis="0 1 0" range="-2.0 2.0"/>
                        <joint name="l_hip_roll" type="hinge" axis="1 0 0" range="-0.5 0.5"/>
                        
                        <body name="left_shin" pos="0 0 -0.18">
                            <geom name="l_shin" type="capsule" size="0.035 0.15" rgba="0.6 0.6 0.6 1"/>
                            <joint name="l_knee" type="hinge" axis="0 1 0" range="0 2.5"/>
                            
                            <body name="left_foot" pos="0 0 -0.15">
                                <geom name="l_foot" type="box" size="0.08 0.04 0.02" rgba="0.5 0.5 0.5 1"/>
                                <joint name="l_ankle" type="hinge" axis="0 1 0" range="-0.5 0.5"/>
                            </body>
                        </body>
                    </body>
                </body>
            </worldbody>
            
            <actuator>
                <!-- Right arm actuators -->
                <motor name="r_shoulder_pitch_motor" joint="r_shoulder_pitch" ctrlrange="-50 50"/>
                <motor name="r_shoulder_roll_motor" joint="r_shoulder_roll" ctrlrange="-50 50"/>
                <motor name="r_elbow_motor" joint="r_elbow" ctrlrange="-50 50"/>
                <motor name="r_wrist_motor" joint="r_wrist" ctrlrange="-20 20"/>
                <motor name="r_finger1_motor" joint="r_finger1" ctrlrange="-5 5"/>
                <motor name="r_finger2_motor" joint="r_finger2" ctrlrange="-5 5"/>
                
                <!-- Left arm actuators -->
                <motor name="l_shoulder_pitch_motor" joint="l_shoulder_pitch" ctrlrange="-50 50"/>
                <motor name="l_shoulder_roll_motor" joint="l_shoulder_roll" ctrlrange="-50 50"/>
                <motor name="l_elbow_motor" joint="l_elbow" ctrlrange="-50 50"/>
                <motor name="l_wrist_motor" joint="l_wrist" ctrlrange="-20 20"/>
                
                <!-- Leg actuators -->
                <motor name="r_hip_pitch_motor" joint="r_hip_pitch" ctrlrange="-100 100"/>
                <motor name="r_hip_roll_motor" joint="r_hip_roll" ctrlrange="-100 100"/>
                <motor name="r_knee_motor" joint="r_knee" ctrlrange="-100 100"/>
                <motor name="r_ankle_motor" joint="r_ankle" ctrlrange="-50 50"/>
                
                <motor name="l_hip_pitch_motor" joint="l_hip_pitch" ctrlrange="-100 100"/>
                <motor name="l_hip_roll_motor" joint="l_hip_roll" ctrlrange="-100 100"/>
                <motor name="l_knee_motor" joint="l_knee" ctrlrange="-100 100"/>
                <motor name="l_ankle_motor" joint="l_ankle" ctrlrange="-50 50"/>
            </actuator>
        </mujoco>
        '''
        return xml
    
    def _get_obs_dim(self):
        """Calculate observation dimension"""
        # Joint positions + velocities + object pose + gripper state
        n_joints = self.model.nq - 7  # Exclude free joint (7 DOF)
        n_joint_vels = self.model.nv - 6  # Exclude free joint velocities
        object_pose = 7  # position (3) + quaternion (4)
        gripper_state = 2  # finger positions
        return n_joints + n_joint_vels + object_pose + gripper_state
    
    def _get_action_dim(self):
        """Calculate action dimension - right arm + gripper actuators"""
        return 6  # r_shoulder_pitch, r_shoulder_roll, r_elbow, r_wrist, r_finger1, r_finger2
    
    def get_observation(self):
        """Get current observation state"""
        # Joint angles (excluding free joint)
        joint_pos = self.data.qpos[7:]  # Skip free joint (7 DOF)
        joint_vel = self.data.qvel[6:]  # Skip free joint velocities (6 DOF)
        
        # Object pose
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        obj_pos = self.data.xpos[obj_id]
        obj_quat = self.data.xquat[obj_id]
        
        # Gripper state (finger positions)
        r_finger1_pos = self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'r_finger1')]
        r_finger2_pos = self.data.qpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'r_finger2')]
        
        obs = np.concatenate([
            joint_pos,
            joint_vel,
            obj_pos,
            obj_quat,
            [r_finger1_pos, r_finger2_pos]
        ])
        
        return obs
    
    def compute_reward(self):
        """Compute reward based on distance reduction + grasp success + placement accuracy"""
        # Get hand position
        hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
        hand_pos = self.data.xpos[hand_id]
        
        # Get object position
        obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
        obj_pos = self.data.xpos[obj_id]
        
        # Distance to object
        dist_to_obj = np.linalg.norm(hand_pos - obj_pos)
        
        # Distance reduction reward
        dist_reward = -dist_to_obj
        
        # Grasp success (simple proximity check)
        grasp_reward = 0.0
        if dist_to_obj < 0.1:  # Close to object
            grasp_reward = 10.0
        
        # Placement accuracy (distance from object to place target)
        place_dist = np.linalg.norm(obj_pos - self.place_target_pos)
        place_reward = -place_dist * 0.5
        
        # Total reward
        total_reward = dist_reward + grasp_reward + place_reward
        
        return total_reward, {
            'distance_to_object': dist_to_obj,
            'grasp_success': grasp_reward > 0,
            'placement_distance': place_dist
        }
    
    def step(self, action):
        """Execute action and return next observation, reward, done, info"""
        # Apply action to right arm and gripper actuators
        actuator_names = [
            'r_shoulder_pitch_motor', 'r_shoulder_roll_motor', 
            'r_elbow_motor', 'r_wrist_motor',
            'r_finger1_motor', 'r_finger2_motor'
        ]
        
        for i, name in enumerate(actuator_names):
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.data.ctrl[actuator_id] = action[i]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get new observation
        obs = self.get_observation()
        
        # Compute reward
        reward, info = self.compute_reward()
        
        # Check if episode is done
        self.episode_step += 1
        done = self.episode_step >= self.max_episode_steps
        
        return obs, reward, done, info
    
    def reset(self):
        """Reset environment to initial state"""
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial pose (standing)
        self.data.qpos[2] = 1.0  # Torso height
        
        # Reset episode counter
        self.episode_step = 0
        
        # Forward simulation to stabilize
        mujoco.mj_forward(self.model, self.data)
        
        return self.get_observation()
    
    def render(self, viewer=None):
        """Render current state with MuJoCo viewer"""
        if viewer is not None:
            viewer.sync()
    
    def demo_episode_with_viewer(self):
        """Run demonstration with visual feedback"""
        obs = self.reset()
        total_reward = 0
        
        print("Starting Valkyrie Pick and Place Demo with Visualization")
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Target object at: {self.target_object_pos}")
        print(f"Place target at: {self.place_target_pos}")
        
        # Create viewer
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Set camera position for better view
            viewer.cam.distance = 2.0
            viewer.cam.azimuth = 45
            viewer.cam.elevation = -15
            
            for step in range(500):  # Extended for better visualization
                # Get current hand and object positions
                hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
                hand_pos = self.data.xpos[hand_id]
                obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
                obj_pos = self.data.xpos[obj_id]
                
                # Simple proportional control towards object
                direction = obj_pos - hand_pos
                distance = np.linalg.norm(direction)
                
                # Scale actions based on distance
                scale = min(1.0, distance * 2.0)
                
                action = np.array([
                    direction[0] * scale,      # r_shoulder_pitch
                    -direction[1] * scale,     # r_shoulder_roll  
                    direction[2] * scale,      # r_elbow
                    0.0,                       # r_wrist
                    0.3 if distance < 0.15 else -0.3,  # r_finger1 (close when near)
                    -0.3 if distance < 0.15 else 0.3   # r_finger2 (close when near)
                ])
                
                # Clip actions to reasonable range
                action = np.clip(action, -1.0, 1.0)
                
                obs, reward, done, info = self.step(action)
                total_reward += reward
                
                # Update viewer
                viewer.sync()
                
                # Control simulation speed
                time.sleep(0.01)
                
                if step % 100 == 0:
                    print(f"Step {step}: Reward={reward:.3f}, Distance={info['distance_to_object']:.3f}, Grasp={info['grasp_success']}")
                
                if done:
                    break
            
            print(f"Episode finished. Total reward: {total_reward:.3f}")
            print("Press any key to close...")
            input()
        
        return total_reward
    
    def demo_episode(self):
        """Run a demonstration episode with simple reaching behavior"""
        obs = self.reset()
        total_reward = 0
        
        print("Starting Valkyrie Pick and Place Demo")
        print(f"Observation dimension: {self.obs_dim}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Target object at: {self.target_object_pos}")
        print(f"Place target at: {self.place_target_pos}")
        
        for step in range(200):
            # Get current hand and object positions
            hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'right_hand')
            hand_pos = self.data.xpos[hand_id]
            obj_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_object')
            obj_pos = self.data.xpos[obj_id]
            
            # Simple proportional control towards object
            direction = obj_pos - hand_pos
            distance = np.linalg.norm(direction)
            
            # Scale actions based on distance
            scale = min(1.0, distance * 2.0)
            
            action = np.array([
                direction[0] * scale,      # r_shoulder_pitch
                -direction[1] * scale,     # r_shoulder_roll  
                direction[2] * scale,      # r_elbow
                0.0,                       # r_wrist
                0.3 if distance < 0.15 else -0.3,  # r_finger1 (close when near)
                -0.3 if distance < 0.15 else 0.3   # r_finger2 (close when near)
            ])
            
            # Clip actions to reasonable range
            action = np.clip(action, -1.0, 1.0)
            
            obs, reward, done, info = self.step(action)
            total_reward += reward
            
            if step % 50 == 0:
                print(f"Step {step}: Reward={reward:.3f}, Distance={info['distance_to_object']:.3f}, Grasp={info['grasp_success']}")
            
            if done:
                break
        
        print(f"Episode finished. Total reward: {total_reward:.3f}")
        return total_reward

def main():
    """Main function to run the POC"""
    try:
        # Create environment
        env = ValkyriePickPlacePOC()
        
        # Choose demo type
        print("Choose demo type:")
        print("1. Text-only demo")
        print("2. Visual demo (with MuJoCo viewer)")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "2":
            # Run visual demonstration
            total_reward = env.demo_episode_with_viewer()
        else:
            # Run text demonstration
            total_reward = env.demo_episode()
        
        print(f"\nPOC completed successfully!")
        print(f"Environment specs:")
        print(f"- Observation space: {env.obs_dim}")
        print(f"- Action space: {env.action_dim}")
        print(f"- Total demo reward: {total_reward:.3f}")
        
    except Exception as e:
        print(f"Error running POC: {e}")
        print("Make sure MuJoCo is properly installed")
        print("For visualization, ensure you have display/GUI support")

if __name__ == "__main__":
    main()