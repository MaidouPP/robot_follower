import tensorflow as tf
import numpy as np
import utils
import time

# How fast is learning
LEARNING_RATE = 0.00001

# How fast does the target net track
TARGET_DECAY = 0.999

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
            self.map_input = tf.placeholder(tf.float32,
                                            [None, self.state_dim],
                                            name="map_input")
            self.action_input = tf.placeholder(tf.float32, [None, 2])
            self.is_training = tf.placeholder(tf.bool, name='is_training')
            self.action_output = self.create_network()

            # Get all the variables in the actor network for exponential moving average, create ema op
            with tf.variable_scope("actor") as scope:
                self.actor_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            self.ema_obj = tf.train.ExponentialMovingAverage(decay=TARGET_DECAY)
            self.compute_ema = self.ema_obj.apply(self.actor_variables)

            # Create target actor network
            self.map_input_target = tf.placeholder(tf.float32,
                                                   [None, self.state_dim],
                                                   name="map_input_target")
            self.action_input_target = tf.placeholder(tf.float32, [None, 2])
            self.action_output_target = self.create_target_network()

            with tf.variable_scope("actor_target") as scope:
                self.actor_target_variables = tf.get_collection(tf.GraphKeys.VARIABLES, scope=scope.name)

            self.actor_target_ema = [tf.assign(self.actor_target_variables[i], \
                                     self.ema_obj.average(self.actor_variables[i]))
                                     for i in xrange(len(self.actor_variables))]


            # Define the gradient operation that delivers the gradients with the action gradient from the critic
            self.q_gradient_input = tf.placeholder(tf.float32, [None, action_size])

            # tf.gradients(
            #     ys,
            #     xs,
            #     grad_ys=None ...)
            # this method computes the gradients of ys w.r.t. xs, and initialized with
            # grad_ys. If grad_ys is none, it's initialized as 1's.
            # Here the gradient is multiplied with q_gradient_input
            self.parameters_gradients = tf.gradients(self.action_output,
                                                     self.actor_variables,
                                                     -self.q_gradient_input)

            # Define the optimizer
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE,
                                                    epsilon=0.01).apply_gradients(
                                                        zip(self.parameters_gradients,
                                                            self.actor_variables))

            # Variables for plotting
            self.actions_mean_plot = [0, 0]
            self.target_actions_mean_plot = [0, 0]

            self.train_counter = 0

    def create_network(self):

        with tf.variable_scope('actor'):

            with tf.variable_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(self.map_input, 128,
                                                        activation_fn=tf.nn.relu,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("fc2"):
                out = tf.contrib.layers.fully_connected(fc1, 2,
                                                        activation_fn=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer())
                out = tf.tanh(out)

            return out

    def create_target_network(self):

        with tf.variable_scope('actor_target'):

            with tf.variable_scope("fc1"):
                fc1 = tf.contrib.layers.fully_connected(self.map_input_target, 128,
                                                        activation_fn=tf.nn.relu,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer())

            with tf.variable_scope("fc2"):
                out = tf.contrib.layers.fully_connected(fc1, 2,
                                                        activation_fn=None,
                                                        biases_initializer=tf.contrib.layers.xavier_initializer())
                out = tf.tanh(out)

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
            self.q_gradient_input: q_gradient_batch,
            self.map_input: state_batch,
            self.action_input: action_batch,
            self.is_training: True})

        # Update the target
        self.update_target()

        self.train_counter += 1

    def update_target(self):

        self.sess.run(self.compute_ema)
        for i in xrange(len(self.actor_variables)):
            self.sess.run(self.actor_target_ema[i])
            # tf.assign(self.actor_target_variables[i], \
            #           self.ema_obj.average(self.actor_variables[i]))

    def get_action(self, state, old_action):

        state = np.expand_dims(state, axis=0)
        return self.sess.run(self.action_output, feed_dict={
            self.map_input: state,
            self.action_input: old_action,
            self.is_training: False})

    def evaluate(self, state_batch, action_batch):

        # Get an action batch
        actions = self.sess.run(self.action_output,
                                feed_dict={self.map_input: state_batch,
                                           self.action_input: action_batch,
                                           self.is_training: False})

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

    def target_evaluate(self, state_batch, action_batch):

        # Get action batch
        actions = self.sess.run(self.action_output_target,
                                feed_dict={self.map_input_target: state_batch,
                                           self.action_input_target: action_batch,
                                           self.is_training: False})

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

        return actions
