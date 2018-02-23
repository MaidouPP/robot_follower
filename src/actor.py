import tensorflow as tf
import numpy as np
import utils
from critic import create_variable
from critic import create_variable_final

# How fast is learning
LEARNING_RATE = 0.0005

# How fast does the target net track
TARGET_DECAY = 0.9999

# How often do we plot variables during training
PLOT_STEP = 10


class ActorNetwork:

    def __init__(self, state_dim, action_size, depth, session, summary_writer):

        self.graph = session.graph

        with self.graph.as_default():

            # Get session and summary writer from ddpg
            self.sess = session
            self.summary_writer = summary_writer

            # Get input dimensions from ddpg
            self.state_dim = state_dim
            self.action_size = action_size
            self.depth = depth

            # # Calculate the fully connected layer size
            # height_layer1 = (state_dim - RECEPTIVE_FIELD1)/STRIDE1 + 1
            # height_layer2 = (height_layer1 - RECEPTIVE_FIELD2)/STRIDE2 + 1
            # height_layer3 = (height_layer2 - RECEPTIVE_FIELD3)/STRIDE3 + 1
            # # height_layer4 = (height_layer3 - RECEPTIVE_FIELD4)/STRIDE4 + 1
            # # self.fully_size = (height_layer4**2) * FILTER4
            # self.fully_size = (height_layer3**2) * FILTER3

            # Create actor network
            self.map_input = tf.placeholder("float", [None, 1, self.state_dim, self.depth])
            self.action_input = tf.placeholder("float", [None, 2])
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.action_output = self.create_network()

            # Get all the variables in the actor network for exponential moving average, create ema op
            with tf.variable_scope("actor") as scope:
                self.actor_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
            self.compute_ema = self.ema_obj.apply(self.actor_variables)

            # Create target actor network
            self.map_input_target = tf.placeholder("float", [None, 1, self.state_dim, self.depth])
            self.action_output_target = self.create_target_network()

            with tf.variable_scope("actor_target") as scope:
                self.actor_target_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            # Define the gradient operation that delivers the gradients with the action gradient from the critic
            self.q_gradient_input = tf.placeholder("float", [None, action_size])
            self.parameters_gradients = tf.gradients(self.action_output, self.actor_variables, -self.q_gradient_input)

            # Define the optimizer
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(zip(self.parameters_gradients,
                                                                                       self.actor_variables))

            # Variables for plotting
            self.actions_mean_plot = [0, 0]
            self.target_actions_mean_plot = [0, 0]

            self.train_counter = 0

    def create_network(self):

        with tf.variable_scope('actor'):

            conv1 = utils.conv(self.map_input, [1, 7, self.depth, 64], [1, 3])
            conv1 = tf.nn.max_pool(conv1,
                                   ksize=[1, 1, 3, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')

            resnet = utils.resnet_block(conv1, [1, 3, conv1.get_shape()[-1], 64], self.is_training)
            resnet = tf.nn.avg_pool(resnet,
                                    ksize=[1, 1, 3, 1],
                                    strides=[1, 1, 3, 1],
                                    padding='SAME')

            fc = tf.reshape(resnet, [resnet.get_shape()[0], -1])
            fc = tf.concat([fc, self.action_input], axis=1)

            fc1 = tf.contrib.layers.fully_connected(fc, 1024)
            fc2 = tf.contrib.layers.fully_connected(fc1, 1024)
            fc3 = tf.contrib.layers.fully_connected(fc2, 512)

            out = tf.contrib.layers.fully_connected(fc3, 2)

            return out


    def create_target_network(self):

        with tf.variable_scope('actor_target'):

            conv1 = utils.conv(self.map_input, [1, 7, self.depth, 64], [1, 3])
            conv1 = tf.nn.max_pool(conv1,
                                   ksize=[1, 1, 3, 1],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME')

            resnet = utils.resnet_block(conv1, [1, 3, conv1.get_shape()[-1], 64], self.is_training)
            resnet = tf.nn.avg_pool(resnet,
                                    ksize=[1, 1, 3, 1],
                                    strides=[1, 1, 3, 1],
                                    padding='SAME')

            fc = tf.reshape(resnet, [resnet.get_shape()[0], -1])
            fc = tf.concat([fc, self.action_input], axis=1)

            fc1 = tf.contrib.layers.fully_connected(fc, 1024)
            fc2 = tf.contrib.layers.fully_connected(fc1, 1024)
            fc3 = tf.contrib.layers.fully_connected(fc2, 512)

            out = tf.contrib.layers.fully_connected(fc3, 2)

            return out


    def restore_pretrained_weights(self, filter_path):

        # First restore the actor net
        saver = tf.train.Saver({"weights_conv1": self.actor_variables[0],
                                "biases_conv1":  self.actor_variables[1],
                                "weights_conv2": self.actor_variables[2],
                                "biases_conv2":  self.actor_variables[3],
                                "weights_conv3": self.actor_variables[4],
                                "biases_conv3":  self.actor_variables[5],
                                # "weights_conv4": self.actor_variables[6],
                                # "biases_conv4":  self.actor_variables[7]
                                })

        saver.restore(self.sess, filter_path)

        # Now restore the target net with
        saver_target = tf.train.Saver({"weights_conv1": self.ema_obj.average(self.actor_variables[0]),
                                       "biases_conv1":  self.ema_obj.average(self.actor_variables[1]),
                                       "weights_conv2": self.ema_obj.average(self.actor_variables[2]),
                                       "biases_conv2":  self.ema_obj.average(self.actor_variables[3]),
                                       "weights_conv3": self.ema_obj.average(self.actor_variables[4]),
                                       "biases_conv3":  self.ema_obj.average(self.actor_variables[5]),
                                       # "weights_conv4": self.ema_obj.average(self.actor_variables[6]),
                                       # "biases_conv4":  self.ema_obj.average(self.actor_variables[7])
                                       })

        saver_target.restore(self.sess, filter_path)

    def train(self, q_gradient_batch, state_batch, action_batch):

        # Train the actor net
        self.sess.run(self.optimizer, feed_dict={
            self.action_input: old_action,
            self.q_gradient_input: q_gradient_batch,
            self.map_input: state_batch,
            self.action_input: action_batch})

        # Update the target
        self.update_target()

        self.train_counter += 1

    def update_target(self):

        sess.run(self.compute_ema)
        self.actor_target_variables = \
            [self.actor_target_variables[i].assign \
             (self.compute_ema.average(self.actor_variables[i]))]

    def get_action(self, state):

        return self.sess.run(self.action_output, feed_dict={self.map_input: [state]})[0]

    def evaluate(self, state_batch):

        # Get an action batch
        actions = self.sess.run(self.action_output, feed_dict={self.map_input: state_batch})

        # Create summaries for the actions
        actions_mean = np.mean(np.asarray(actions, dtype=float), axis=0)
        self.actions_mean_plot += actions_mean

        # Only save files every PLOT_STEP steps
        if self.train_counter % PLOT_STEP == 0:

            self.actions_mean_plot /= PLOT_STEP

            summary_action_0 = tf.Summary(value=[tf.Summary.Value(tag='actions_mean[0]',
                                                                  simple_value=np.asscalar(
                                                                      self.actions_mean_plot[0]))])
            summary_action_1 = tf.Summary(value=[tf.Summary.Value(tag='actions_mean[1]',
                                                                  simple_value=np.asscalar(
                                                                      self.actions_mean_plot[1]))])
            self.summary_writer.add_summary(summary_action_0, self.train_counter)
            self.summary_writer.add_summary(summary_action_1, self.train_counter)

            self.actions_mean_plot = [0, 0]

        return actions

    def target_evaluate(self, state_batch):

        # Get action batch
        actions = self.sess.run(self.action_output_target, feed_dict={self.map_input_target: state_batch})

        # Create summaries for the target actions
        actions_mean = np.mean(np.asarray(actions, dtype=float), axis=0)
        self.target_actions_mean_plot += actions_mean

        # Only save files every 10 steps
        if (self.train_counter % PLOT_STEP) == 0:

            self.target_actions_mean_plot /= PLOT_STEP

            summary_target_action_0 = tf.Summary(value=[tf.Summary.Value(tag='target_actions_mean[0]',
                                                                         simple_value=np.asscalar(
                                                                             self.target_actions_mean_plot[0]))])
            summary_target_action_1 = tf.Summary(value=[tf.Summary.Value(tag='target_actions_mean[1]',
                                                                         simple_value=np.asscalar(
                                                                             self.target_actions_mean_plot[1]))])
            self.summary_writer.add_summary(summary_target_action_0, self.train_counter)
            self.summary_writer.add_summary(summary_target_action_1, self.train_counter)

            self.target_actions_mean_plot = [0, 0]

        return action
