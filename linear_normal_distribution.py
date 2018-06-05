import tensorflow as tf
import numpy as np

MEAN = 3
STDDEV = 2
BATCH_SIZE = 100
HIDDEN_LAYERS_NUMBER = 3
HIDDEN_LAYER_SIZE = 64
DISCRIMINATOR_OUTPUT_SIZE = 1

def data_batch(mean, stddev, batch_size):
    return tf.random_normal([batch_size, 1], mean, stddev)


def generator_input(batch_size):
    return tf.random_normal([batch_size, 1], 0., 1.)


def generator(input_data):
    mean = tf.Variable(tf.constant(0.))
    stddev = tf.Variable(tf.constant(1.))
    return input_data * stddev + mean


def discriminator(input_data):
    result = input_data
    features = 1
    for i in range(HIDDEN_LAYERS_NUMBER):
        weights = tf.get_variable(
            "weights_%d" % i, 
            initializer=tf.truncated_normal([features, HIDDEN_LAYER_SIZE], stddev=0.1)
        )
        biases = tf.get_variable(
            "biases_%d" % i,
            initializer=tf.constant(0.1, shape=[HIDDEN_LAYER_SIZE])
        )
        result = tf.nn.relu(tf.matmul(result, weights) + biases)
        features = HIDDEN_LAYER_SIZE
    
    weights = tf.get_variable(
        "weights_out", 
        initializer=tf.truncated_normal([features, DISCRIMINATOR_OUTPUT_SIZE], stddev=0.1)
    )
    biases = tf.get_variable(
        "biases_out",
        initializer=tf.constant(0.1, shape=[DISCRIMINATOR_OUTPUT_SIZE])
    )
    return tf.matmul(result, weights) + biases


def model():
    generated = generator(generator_input(BATCH_SIZE))
    real = data_batch(MEAN, STDDEV, BATCH_SIZE)
    with tf.variable_scope('discriminator'):
        out_for_real = discriminator(real)
    with tf.variable_scope('discriminator', reuse=True):
        out_for_generaded = discriminator(generated)

    loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(out_for_real),
            logits=out_for_real
        )
    )

    loss_generated = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(out_for_generaded),
            logits=out_for_generaded
        )
    )

    discriminator_loss = loss_real + loss_generated

    generator_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(out_for_generaded),
            logits=out_for_generaded
        )
    )

    generator_train = tf.train.AdamOptimizer().minimize(
        generator_loss, name="train_generator"
    )

    discriminator_train = tf.train.AdamOptimizer().minimize(
        discriminator_loss, name="train_discriminator"
    )


with tf.Session() as sess:
    writer = tf.summary.FileWriter('.')
    writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    sess.run(model())
