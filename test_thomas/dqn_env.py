import gym

from gym_super_mario_bros import SuperMarioBrosEnv

class CustomEnv(SuperMarioBrosEnv):

    @property
    def _test(self):
        return "Coucou Clément"

    def _custom_reward(self):
        return 10

    def _get_reward(self):
        return self._custom_reward()

    def _custom_state(self):
        return self.ram

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._custom_state(), reward, done, info

    def _get_info(self):
        parent_info = super()._get_info()
        parent_info["test"] = self._test
        return parent_info

    def action_spec(self):
        return None

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