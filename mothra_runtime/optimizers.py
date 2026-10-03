"""Training optimizers with the historical embedding-gradient semantics."""
import keras
import tensorflow as tf


@keras.saving.register_keras_serializable(package="Mothra")
class DeduplicatingAdam(keras.optimizers.Adam):
    """Sum repeated embedding indices before Adam's second-moment update.

    Keras 2's OptimizerV2 aggregates duplicate IndexedSlices before squaring
    them. Converting the small (64 x 64) embedding gradient to a dense tensor
    preserves that ordering, including decay of moments for unvisited rows.
    """

    def update_step(self, gradient, variable, learning_rate):
        if isinstance(gradient, tf.IndexedSlices):
            gradient = tf.convert_to_tensor(gradient)
        super().update_step(gradient, variable, learning_rate)
