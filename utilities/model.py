import tensorflow as tf
from utilities.config import *



def discriminator(x):
    splits = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1))(x)
    state_0 = tf.keras.layers.Dense(Config.n_hidden_units_d)(splits[0])    
    state_0 = tf.keras.layers.Activation(lambda x: tf.keras.activations.relu(x, alpha=Config.alpha))(state_0)
    state_0 = tf.keras.layers.Dense(Config.n_hidden_units_d)(state_0)
    state_1 = tf.keras.layers.Dense(Config.n_hidden_units_d)(splits[0])
    state_1 = tf.keras.layers.Activation(lambda x: tf.keras.activations.relu(x, alpha=Config.alpha))(state_1)
    state_1 = tf.keras.layers.Dense(Config.n_hidden_units_d)(state_1)
    x = tf.compat.v1.keras.layers.CuDNNLSTM(Config.n_hidden_units_d, return_sequences=True)(x, initial_state = [state_0,state_1])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation = "sigmoid"))(x)
    return x

def generator(x):
    splits = tf.keras.layers.Lambda(lambda x: tf.unstack(x, axis=1))(x)
    state_0 = tf.keras.layers.Dense(Config.n_hidden_units_d)(splits[0])
    state_0 = tf.keras.layers.Activation(lambda x: tf.keras.activations.relu(x, alpha=Config.alpha))(state_0)
    state_0 = tf.keras.layers.Dense(Config.n_hidden_units_d)(state_0)
    state_1 = tf.keras.layers.Dense(Config.n_hidden_units_d)(splits[0])
    state_1 = tf.keras.layers.Activation(lambda x: tf.keras.activations.relu(x, alpha=Config.alpha))(state_1)
    state_1 = tf.keras.layers.Dense(Config.n_hidden_units_d)(state_1)
    x = tf.compat.v1.keras.layers.CuDNNLSTM(Config.n_hidden_units_g, return_sequences=True)(x, initial_state = [state_0,state_1])
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1))(x)
    return x

