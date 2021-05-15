import tensorflow.compat.v1 as tf
import gym_super_mario_bros as gym_smb
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

from nsmm3dq.q_learning import QLearning
from nsmm3dq.wrappers import (
    MarioRamWrapper,
    SkipEnv,
    CustomReward,
)
from nsmm3dq.agents.mario_ram import MarioRam

tf.disable_v2_behavior()

if gpus := tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpus[0], True)

env = gym_smb.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, RIGHT_ONLY)
env = MarioRamWrapper(env)
env = SkipEnv(env, 4)
env = CustomReward(env)

learner = QLearning(env, MarioRam)
learner.start()
