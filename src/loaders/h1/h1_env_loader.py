

from typing import Dict
from mujoco_playground._src import mjx_env
from envs.h1_env import H1InPlaceEnv
from configs.constants.h1 import H1Constants

from configs.default_config import default_config
from loaders.training_env_loader import TrainingEnvLoader
from loaders.go1.go1_constants import Go1Constants
from configs.brax_ppo_config import brax_ppo_config
from randomizers.domain_randomize import domain_randomize


class H1Loader(TrainingEnvLoader):

  def set_model_constants(self):
    return H1Constants

  def get_assets(self) -> Dict[str, bytes]:
    assets = {}
    xml_path = self.model_constants.XML_ROOT_PATH
    assets_path = xml_path / "assets"

    menagerie_xml_path = self.model_constants.MENAGERIE_XML_PATH
    menagerie_assets_path = menagerie_xml_path / "assets"

    mjx_env.update_assets(assets, xml_path, "*.xml")
    mjx_env.update_assets(assets, assets_path)

    mjx_env.update_assets(assets, menagerie_xml_path, "*.xml")
    mjx_env.update_assets(assets, menagerie_assets_path)
    return assets


  def set_default_config(self):
    return default_config()

  def get_brax_ppo_config(self):
    return brax_ppo_config(self.config)

  def get_randomizer(self):
    return domain_randomize

  def get_env(self):
    print("self.model_constants", self.model_constants)
    return H1InPlaceEnv(
        self.model_constants,
        get_assets=self.get_assets,
        task="flat_terrain",
        config=self.config,
    )




h1_loader = H1Loader()
h1_env = h1_loader.get_env()
ppo_params = h1_loader.get_brax_ppo_config()


# env_name = 'Go1JoystickFlatTerrain'
# env_cfg_ref = registry.get_default_config(env_name)
# env_ref = registry.load(env_name)