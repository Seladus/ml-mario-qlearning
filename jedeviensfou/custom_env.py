import gym

from gym_super_mario_bros import SuperMarioBrosEnv

ENEMY_DRAWN = 0x000F
ENEMY_POSITION_LEVEL = 0x006E
ENEMY_POSITION_SCREEN = 0x0087


class CustomEnv(SuperMarioBrosEnv):
    def _custom_state(self):
        info = self._get_info()
        return [info["x_pos"], info["y_pos"]] + [
            self._enemy_position(i) for i in range(5)
        ]

    def step(self, action):
        _, reward, done, info = super().step(action)
        return self._custom_state(), reward, done, info

    def _enemy_position(self, i):
        if i > 4:
            return -1
        if not self.ram[ENEMY_DRAWN + i]:
            return 0
        return (
            self.ram[ENEMY_POSITION_LEVEL + i] * 0x100
            + self.ram[ENEMY_POSITION_SCREEN + i]
        )

    def reset(self):
        _ = super().reset()
        return self._custom_state()


def register_custom_env():

    ENV_NAME = "CustomEnv-v0"

    gym.envs.registration.register(
        id=ENV_NAME,
        entry_point="custom_env:CustomEnv",
        max_episode_steps=9999999,
        reward_threshold=9999999,
        nondeterministic=True,
    )


register_custom_env()
