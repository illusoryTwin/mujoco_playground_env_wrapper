import functools
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground import registry
from mujoco_playground import wrapper
from src.configs.go1.handstand_default_config import go1_handstand_default_config
from src.configs.go1.handstand_default_config import go1_handstand_default_config
from src.brax_ppo_configs.go1.handstand_brax_ppo_config import go1_handstand_brax_ppo_config
from src.randomizers.go1.go1_randomizer import domain_randomize
from src.configs.constants.go1_constants import Go1Constants
from src.loaders.training_env_loader import TrainingEnvLoader
from src.environments.go1.go1_handstand_env import Go1HandstandEnv
from brax.training.agents import ppo
from src.utils.progress import progress
from datetime import datetime


go1_handstand_default_config = go1_handstand_default_config 
go1_handstand_brax_ppo_config = go1_handstand_brax_ppo_config
go1_handstand_randomizer = domain_randomize  # `randomizer` refers to the function directly
go1_handstand_constants = Go1Constants


go1_handstand_loader = TrainingEnvLoader(Go1HandstandEnv,
                                          go1_handstand_constants,
                                          go1_handstand_default_config,
                                          go1_handstand_brax_ppo_config,
                                          go1_handstand_randomizer) 
go1_handstand_env = go1_handstand_loader.get_env()
ppo_params = go1_handstand_loader.get_brax_ppo_config()

env_name = 'Go1Handstand'
env_cfg_ref = registry.get_default_config(env_name)
env_ref = registry.load(env_name)

times = [datetime.now()]

ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=go1_handstand_randomizer, # randomizer,
    progress_fn=progress
)

go1_handstand_env_name_ref = 'Go1Handstand'
go1_handstand_env_ref = registry.load(go1_handstand_env_name_ref)
go1_handstand_env_cfg_ref = registry.get_default_config('Go1Handstand') # go1_handstand_env_name_ref)

make_inference_fn, params, metrics = train_fn(
    environment=go1_handstand_env, 
    eval_env=registry.load(go1_handstand_env_name_ref, config=go1_handstand_env_cfg_ref),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)

print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")