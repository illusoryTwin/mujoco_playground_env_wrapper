from abc import ABC, abstractmethod
from typing import Dict, Type
from mujoco_playground._src import mjx_env


class BaseTrainingEnvLoader(ABC):
  """
  Abstract base class for loading model constants, configuration, and assets
  for further usage while training.
  """
  def __init__(self):
    self.config = self.set_default_config()
    self.model_constants = self.set_model_constants()

  @abstractmethod
  def set_model_constants(self) -> Type:
    pass


  @abstractmethod
  def get_assets(self) -> Dict[str, bytes]:
    pass


  @abstractmethod
  def set_default_config(self):
    pass


  @abstractmethod
  def get_brax_ppo_config(self):
    pass

  @abstractmethod
  def get_randomizer(self):
    pass

  @abstractmethod
  def get_env(self):
    pass



class TrainingEnvLoader(BaseTrainingEnvLoader):

  def __init__(self,
               env,
               constants,
               config_fn,
               brax_ppo_config_fn,
               randomizer_fn
               ):
    self.env = env
    self.model_constants = constants or self.set_model_constants()
    self.config_fn = config_fn or self.set_default_config()
    self.brax_ppo_config_fn = brax_ppo_config_fn
    self.randomizer_fn = randomizer_fn

    super().__init__()


  def set_model_constants(self) -> Type:
    return self.model_constants  # Implement abstract method

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
    return self.config_fn()

  def get_brax_ppo_config(self):
    return self.brax_ppo_config_fn(self.config)

  def get_randomizer(self):
    return self.randomizer_fn

  def get_env(self):
    return self.env(
        self.model_constants,
        get_assets=self.get_assets,
        # task="flat_terrain",
        config=self.config,
    )
