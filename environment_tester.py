import argparse

import tensorflow.compat.v1 as tf
import gym_super_mario_bros as gym_smb
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

from nsmm3dq.wrappers import (
    MaxAndSkipEnv,
    WarpFrame,
    FrameStack,
    ScaledFloatFrame,
    CustomReward,
    MarioRamWrapper,
    SkipEnv,
)
from nsmm3dq.agents.mario_conv import MarioConv
from nsmm3dq.agents.mario_ram import MarioRam

from jedeviensfou.wrappers_gym import wrapper

tf.disable_v2_behavior()

if gpus := tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpus[0], True)


parser = argparse.ArgumentParser(description="Test a model on the mario environment.")
parser.add_argument("model", help="The path to the model")
parser.add_argument(
    "-t", "--type", help="The type of model", choices=["conv", "ram"], default="conv"
)
parser.add_argument("-v", "--video", help="The video output", required=False)
args = parser.parse_args()

env = gym_smb.make("SuperMarioBros-1-1-v0")
env = JoypadSpace(env, RIGHT_ONLY)

env_ref = gym_smb.make("SuperMarioBros-1-1-v0")
env_ref = JoypadSpace(env_ref, RIGHT_ONLY)

video_recorder = None
if args.video:
    video_recorder = VideoRecorder(env_ref, args.video, enabled=True)
else:
    video_recorder = VideoRecorder(env_ref, "", enabled=False)

agent_type = None
if args.type == "conv":
    env = MaxAndSkipEnv(env, 4)
    env = WarpFrame(env)
    env = FrameStack(env, 4)
    env = ScaledFloatFrame(env)
    agent_type = MarioConv
else:
    env = MarioRamWrapper(env)
    env = SkipEnv(env)
    agent_type = MarioRam

env = CustomReward(env)
agent = agent_type(
    env.observation_space.shape,
    env.action_space.n,
    demo_mode=True,
    model_path=args.model,
)

skipped_frame = 4
pause_at_end = False

done = False
score = 0
state = env.reset()
env_ref.reset()

while not done:
    action = agent.choose_action(state)
    frame = env.render()
    next_state, reward, done, info = env.step(action=action)
    if not done:
        for _ in range(skipped_frame):
            env_ref.step(action=action)
            video_recorder.capture_frame()
    else:
        env_ref.step(action=action)
        video_recorder.capture_frame()
    score += reward
    state = next_state

env_ref.close()
video_recorder.close()
print(f"score : {score:.2f}")

env.close()
