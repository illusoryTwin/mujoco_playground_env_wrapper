from mujoco_playground._src import mjx_env
from etils import epath
import numpy as np 

class H1Constants:

  XML_ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "h1" / "xmls"
  FEET_ONLY_XML = XML_ROOT_PATH / "scene_mjx_feetonly.xml"

  FEET_SITES = [
      "left_foot",
      "right_foot",
  ]

  LEFT_FEET_GEOMS = [
      "left_foot1",
      "left_foot2",
      "left_foot3",
  ]
  RIGHT_FEET_GEOMS = [
      "right_foot1",
      "right_foot2",
      "right_foot3",
  ]


  _PHASES = np.array([
      [0, np.pi],  # walk
      [0, 0],  # jump
  ])

  ROOT_BODY = "torso_link"

  GRAVITY_SENSOR = "upvector"
  GLOBAL_LINVEL_SENSOR = "global_linvel"
  GLOBAL_ANGVEL_SENSOR = "global_angvel"
  LOCAL_LINVEL_SENSOR = "local_linvel"
  ACCELEROMETER_SENSOR = "accelerometer"
  GYRO_SENSOR = "gyro"


  ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "h1"
  XML_ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "h1" / "xmls"
  MENAGERIE_XML_PATH = mjx_env.MENAGERIE_PATH / "unitree_h1"

  @classmethod
  def task_to_xml(cls, task_name: str) -> epath.Path:
      task_map = {
          "flat_terrain": cls.FEET_ONLY_FLAT_TERRAIN_XML,
          "rough_terrain": cls.FEET_ONLY_ROUGH_TERRAIN_XML,
      }
      if task_name not in task_map:
          raise ValueError(f"Invalid task name: {task_name}. Must be one of {list(task_map.keys())}")
      return task_map[task_name]