import tensorflow.compat.v1 as tf

from nsmm3dq.agent import Agent


class MarioRam(Agent):
    def create_model(self, learning_rate, input_dims, nb_actions):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(input_dims),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(nb_actions, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), loss="huber_loss"
        )
        return model
