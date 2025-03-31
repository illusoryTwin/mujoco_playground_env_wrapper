import numpy as np 
from ml_collections import config_dict 

_PHASES = np.array([
    [0, np.pi],  # walk
    [0, 0],  # jump
])


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=1000,
      early_termination=True,
      action_repeat=1,
      action_scale=0.6,
      history_len=3,
      obs_noise=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Rewards.
              feet_phase=2.0,
              # Costs.
              pose=-0.5,
              ang_vel=-0.5,
              lin_vel=-0.5,
          ),
      ),
      gait_frequency=[0.5, 4.0],
      gaits=["walk", "jump"],
      foot_height=[0.08, 0.4],
  )

