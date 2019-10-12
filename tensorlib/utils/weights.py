import tensorflow as tf


def hard_update(target: tf.keras.Model, source: tf.keras.Model):
    target.set_weights(source.get_weights())


def soft_update(target: tf.keras.Model, source: tf.keras.Model, tau):
    new_weights = []
    for target_weights, source_weights in zip(target.get_weights(), source.get_weights()):
        new_weights.append(target_weights * (1. - tau) + source_weights * tau)
    target.set_weights(new_weights)