import tensorflow.compat.v1 as tf

from nsmm3dq.agent import Agent


class MarioConv(Agent):
    def create_model(self, learning_rate, input_dims, nb_actions):
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(input_dims),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=8, strides=4, activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=4, strides=2, activation="relu"
                ),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, activation="relu"
                ),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(nb_actions, activation="linear"),
            ]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate), loss="huber_loss"
        )
        return model
