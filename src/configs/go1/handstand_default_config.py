from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from etils import epath
import numpy as np

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=500,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.3,
      soft_joint_pos_limit_factor=0.9,
      init_from_crouch=0.0,
      energy_termination_threshold=np.inf,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              height=1.0,
              orientation=1.0,
              contact=-0.1,
              action_rate=0.0,
              termination=0.0,
              dof_pos_limits=-0.5,
              torques=0.0,
              pose=-0.1,
              stay_still=0.0,
              # For finetuning, use energy=-0.003 and dof_acc=-2.5e-7.
              energy=0.0,
              dof_acc=0.0,
          ),
      ),
  )