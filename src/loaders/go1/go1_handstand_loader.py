# from ml_collections import config_dict
# from typing import Dict 
# from mujoco_playground._src import mjx_env
# from mujoco_playground._src import mjx_env


# class Go1HandstandLoader:
# #   def __init__(self):
#   def __init__(self, config, constants, env):

#     self.config = self.get_default_config()
#     self.model_constants = Go1Constants


#   def get_assets(self) -> Dict[str, bytes]:
#     assets = {}
#     xml_path = self.model_constants.XML_ROOT_PATH
#     assets_path = xml_path / "assets"

#     menagerie_xml_path = self.model_constants.MENAGERIE_XML_PATH
#     menagerie_assets_path = menagerie_xml_path / "assets"

#     mjx_env.update_assets(assets, xml_path, "*.xml")
#     mjx_env.update_assets(assets, assets_path)

#     mjx_env.update_assets(assets, menagerie_xml_path, "*.xml")
#     mjx_env.update_assets(assets, menagerie_assets_path)
#     return assets


#   def get_default_config(self):
#     return default_config()

#   def get_brax_ppo_config(self):
#     return brax_ppo_config(self.config)

#   def load_env(self):
#     return Handstand(
#         self.model_constants,
#         get_assets=self.get_assets,
#         config=self.config,
#     )

#   def randomizer(self):
#     return domain_randomize



# go1_handstand_loader = Go1HandstandLoader()
# go1_handstand_env_test = go1_handstand_loader.load_env()
# ppo_params = go1_handstand_loader.get_brax_ppo_config()