import gym

from gym_super_mario_bros import SuperMarioBrosEnv

class CustomEnv(SuperMarioBrosEnv):

    def _custom_state(self):
        return self.ram

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._custom_state(), reward, done, info

    def reset(self):
        _ = super().reset()
        return self._custom_state()

def register_custom_env():
    import gym

    ENV_NAME = "CustomEnv-v0"

    gym.envs.registration.register(
        id=ENV_NAME,
        entry_point="dqn_env:CustomEnv",
        max_episode_steps=9999999,
        reward_threshold=9999999,
        nondeterministic=True
    )

register_custom_env()
