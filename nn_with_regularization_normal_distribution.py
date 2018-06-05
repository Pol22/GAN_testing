import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


HIDDEN_LAYER_SIZE = 256
GEN_HIDDEN_LAYER_SIZE = 256
HIDDEN_LAYERS_NUMBER = 3
GEN_HIDDEN_LAYERS_NUMBER = 3
DISCRIMINATOR_OUTPUT_SIZE = 1
BATCH_SIZE = 200
BETA = 0.01

input_data = tf.placeholder(dtype=tf.float32, shape=(BATCH_SIZE, 1))

def discriminator(input_data):
    result = input_data
    features = 1
    with tf.variable_scope('discriminator_params', reuse=tf.AUTO_REUSE):
        for i in range(HIDDEN_LAYERS_NUMBER):
            weights = tf.get_variable(
                "weights_%d" % i, 
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                shape=[features, HIDDEN_LAYER_SIZE]
            )
            biases = tf.get_variable(
                "biases_%d" % i,
                initializer=tf.constant_initializer(0.1),
                shape=[HIDDEN_LAYER_SIZE]
            )
            result = tf.nn.relu(tf.matmul(result, weights) + biases)
            #result = tf.nn.dropout(result, 0.25)
            features = HIDDEN_LAYER_SIZE
        
        weights = tf.get_variable(
            "weights_out", 
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            shape=[features, DISCRIMINATOR_OUTPUT_SIZE], 
        )
        biases = tf.get_variable(
            "biases_out",
            initializer=tf.constant_initializer(0.1),
            shape=[DISCRIMINATOR_OUTPUT_SIZE]
        )
        return tf.matmul(result, weights) + biases


generator_input = tf.random_normal(shape=(BATCH_SIZE, 1), mean=0.0, stddev=1.0)

def generator(generator_input):
    result = generator_input
    features = 1
    with tf.variable_scope('generator_params', reuse=tf.AUTO_REUSE):
        for i in range(GEN_HIDDEN_LAYERS_NUMBER):
            weights = tf.get_variable(
                "gen_weights_%d" % i, 
                initializer=tf.truncated_normal_initializer(stddev=0.1),
                shape=[features, GEN_HIDDEN_LAYER_SIZE]
            )
            biases = tf.get_variable(
                "gen_biases_%d" % i,
                initializer=tf.constant_initializer(0.1),
                shape=[GEN_HIDDEN_LAYER_SIZE]
            )
            result = tf.nn.relu(tf.matmul(result, weights) + biases)
            result = tf.nn.dropout(result, 0.25)
            features = GEN_HIDDEN_LAYER_SIZE
        
        weights = tf.get_variable(
            "gen_weights_out", 
            initializer=tf.truncated_normal_initializer(stddev=0.1),
            shape=[features, 1], 
        )
        biases = tf.get_variable(
            "gen_biases_out",
            initializer=tf.constant_initializer(0.1),
            shape=[1]
        )
        return tf.matmul(result, weights) + biases
        

generated = generator(generator_input)
out_for_real = discriminator(input_data)
out_for_generated = discriminator(generated)

average_probability_real = tf.reduce_mean(tf.sigmoid(out_for_real))
average_probability_fake = tf.reduce_mean(tf.sigmoid(out_for_generated))

tf.summary.scalar("P_real_on_real", average_probability_real)
tf.summary.scalar("P_real_on_fake", average_probability_fake)

loss_real = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(out_for_real),
        logits=out_for_real
    )
)

loss_generated = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.zeros_like(out_for_generated),
        logits=out_for_generated
    )
)

discriminator_loss = loss_real + loss_generated
# with tf.variable_scope('discriminator_params', reuse=tf.AUTO_REUSE):
#     for i in range(HIDDEN_LAYERS_NUMBER):
#         discriminator_loss = discriminator_loss + BETA * tf.nn.l2_loss(tf.get_variable("weights_%d" % i))
#         discriminator_loss = discriminator_loss + BETA * tf.nn.l2_loss(tf.get_variable("biases_%d" % i))
#     discriminator_loss = discriminator_loss + BETA * tf.nn.l2_loss(tf.get_variable("weights_out"))
#     discriminator_loss = discriminator_loss + BETA * tf.nn.l2_loss(tf.get_variable("biases_out"))

generator_loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.ones_like(out_for_generated),
        logits=out_for_generated
    )
)
# with tf.variable_scope('generator_params', reuse=tf.AUTO_REUSE):
#     for i in range(GEN_HIDDEN_LAYERS_NUMBER):
#         generator_loss = generator_loss + BETA * tf.nn.l2_loss(tf.get_variable("gen_weights_%d" % i))
#         generator_loss = generator_loss + BETA * tf.nn.l2_loss(tf.get_variable("gen_biases_%d" % i))
#     generator_loss = generator_loss + BETA * tf.nn.l2_loss(tf.get_variable("gen_weights_out"))
#     generator_loss = generator_loss + BETA * tf.nn.l2_loss(tf.get_variable("gen_biases_out"))
    

with tf.variable_scope('generator_params') as scope:
    generator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]
        
with tf.variable_scope('discriminator_params') as scope:
    discriminator_variables = [v for v in tf.global_variables() if v.name.startswith(scope.name)]

generator_train = tf.train.AdamOptimizer().minimize(
    generator_loss,
    var_list=generator_variables,
    name='train_generator'
)

discriminator_train = tf.train.AdamOptimizer().minimize(
    discriminator_loss,
    var_list=discriminator_variables,
    name='train_discriminator'
)



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    writer = tf.summary.FileWriter('./log2/', sess.graph)
    merged = tf.summary.merge_all()
    mean1_train = 5.5
    stddev1_train = 3.25
    mean2_train = -3.2
    stddev2_train = 1.75
    for step in range(12001):
        train_data_1 = np.random.normal(mean1_train, stddev1_train, size=(BATCH_SIZE // 2, 1))
        train_data_2 = np.random.normal(mean2_train, stddev2_train, size=(BATCH_SIZE // 2, 1))
        train_data = np.concatenate((train_data_1, train_data_2))
        np.random.shuffle(train_data)
        _, _, _, _, summary = sess.run(
            [average_probability_real,
            average_probability_fake,
            discriminator_train,
            generator_train,
            merged],
            feed_dict={input_data: train_data}
        )
        if step % 10 == 9:
            writer.add_summary(summary, step)
        if step % 100 == 99:
            loss_gen = sess.run(generator_loss)
            loss_discr = sess.run(discriminator_loss, feed_dict={input_data: train_data})
            print('Step:', step, ', gen loss:', loss_gen, ', discr loss:', loss_discr)
    data = []
    num = 5000
    for _ in range(num):
        data.append(sess.run(generated))
    data = np.reshape(data, (BATCH_SIZE * num))
    plt.hist(data, 500, facecolor='g', alpha=0.5, range=(-10, 15))
    real_data_1 = np.random.normal(mean1_train, stddev1_train, size=(BATCH_SIZE * num // 2, 1))
    real_data_2 = np.random.normal(mean2_train, stddev2_train, size=(BATCH_SIZE * num // 2, 1))
    real = np.concatenate((real_data_1, real_data_2))
    real = np.reshape(real, (BATCH_SIZE * num))
    plt.hist(real, 500, facecolor='r', alpha=0.5, range=(-10, 15))
    handles = [Rectangle((0,0),1,1,color='g',ec="k"), Rectangle((0,0),1,1,color='r',ec="k")]
    labels= ['Generator distribution', 'Real distribution']
    plt.legend(handles, labels)
    plt.xlabel('x')
    plt.title('Distributions')
    plt.show()
