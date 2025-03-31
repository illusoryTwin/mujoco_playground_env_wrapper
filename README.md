# Mujoco Playground Env Wrapper

A wrapper for the Mujoco Playground environment.



The `mujoco_playground_env_wrapper` provides a simple interface for loading and interacting with Mujoco-based environments. It allows you to instantiate the `TrainingEnvLoader` class with a set of predefined parameters, including environment configurations, constants, training configurations, and randomizers.


### Example
You can find this example inside `src/loaders/go1/go1_joystick_loader.py`


```
go1_default_config = default_config
go1_brax_ppo_config = brax_ppo_config
go1_randomizer = domain_randomize  
go1_constants = Go1Constants


go1_loader = TrainingEnvLoader(Go1JoystickEnv,
                               go1_constants,
                               go1_default_config,
                               go1_brax_ppo_config,
                               domain_randomize)
go1_js_env_test = go1_loader.get_env()
ppo_params = go1_loader.get_brax_ppo_config()
```


