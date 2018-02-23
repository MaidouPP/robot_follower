#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import time

dir_path = os.path.dirname(os.path.realpath(__file__))

# tensorflow log dir
TF_LOG_PATH = dir_path + '/../log'

# tensorflow output dir
TF_OUT_PATH = dir_path + '/../output'

# experience data path
EXP_PATH = dir_path + '/../exp'


class DDPG:

    def __init__(self,
                 pretrain = False,
                 net_path = None):

        if not tf.gfile.Exists(TF_LOG_DIR):
            tf.gfile.MakeDirs(TF_LOG_DIR)
        if not tf.gfile.Exists(TF_OUT_DIR):
            tf.gfile.MakeDirs(TF_OUT_DIR)

        self.session = tf.session()
        self.graph = self.session.graph()

        with self.graph.as_default():

            self.batch_size = 32

            self.depth = 4
            self.length = 662
            self.action_dim = 2

            self.old_state = np.zeros(
                (self.length, self.depth), dtype='float32')
            self.old_action = np.ones(self.action_dim, dtype='float32')
            self.network_action = np.zeros(2, dtype='float32')
            self.noise_action = np.zeros(2, dtype='float32')
            self.action = np.zeros(2, dtype='float32')

            # Initialize the grad inverter object to keep the action bounds
            self.grad_inv = GradInverter(A0_BOUNDS, A1_BOUNDS, self.session)

            # # Make sure the directory for the data files exists
            # if not tf.gfile.Exists(DATA_PATH):
            #     tf.gfile.MakeDirs(DATA_PATH)

            # Initialize summary writers to plot variables during training
            self.summary_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(TF_LOG_PATH)

            # Initialize actor and critic networks
            self.actor_network = ActorNetwork(self.length,
                                              self.depth,
                                              self.action_dim,
                                              self.session,
                                              self.summary_writer)
            self.critic_network = CriticNetwork(self.length,
                                                self.depth,
                                                self.action_dim,
                                                self.session,
                                                self.summary_writer)

            # Initialize the saver to save the network params
            self.saver = tf.train.Saver()

            # initialize the experience data manger
            self.data_manager = DataManager(self.batch_size, EXP_PATH, self.session)

            # Uncomment if collecting a buffer for the autoencoder
            # self.buffer = deque()

            # Should we load the pre-trained params?
            # If so: Load the full pre-trained net
            # Else:  Initialize all variables the overwrite the conv layers with the pretrained filters
            if pretrain:
                self.saver.restore(self.session, net_path)
            else:
                self.session.run(tf.initialize_all_variables())

            tf.train.start_queue_runners(sess=self.session)
            time.sleep(1)

            # Initialize a random process the Ornstein-Uhlenbeck process for action exploration
            self.exploration_noise = OUNoise(self.action_dim, MU, THETA, SIGMA)
            self.noise_flag = True

            # Initialize time step
            self.training_step = 0

            # Flag: don't learn the first experience
            self.first_experience = True

            # After the graph has been filled add it to the summary writer
            self.summary_writer.add_graph(self.graph)

    def train(self):

        # Check if the buffer is big enough to start training
        if self.data_manager.enough_data():

            # get the next random batch from the data manger
            state_batch, \
                action_batch, \
                reward_batch, \
                next_state_batch, \
                is_episode_finished_batch = self.data_manager.get_next_batch()

            state_batch = np.divide(state_batch, 100.0)
            next_state_batch = np.divide(next_state_batch, 100.0)

            # Are we visualizing the first state batch for debugging?
            # If so: We have to scale up the values for grey scale before plotting
            # if self.visualize_input:
            #     state_batch_np = np.asarray(state_batch)
            #     state_batch_np = np.multiply(state_batch_np, -100.0)
            #     state_batch_np = np.add(state_batch_np, 100.0)
            #     self.viewer.set_data(state_batch_np)
            #     self.viewer.run()
            #     self.visualize_input = False

            # Calculate y for the td_error of the critic
            y_batch = []
            next_action_batch = self.actor_network.target_evaluate(
                next_state_batch)
            q_value_batch = self.critic_network.target_evaluate(
                next_state_batch, next_action_batch)

            for i in range(0, self.batch_size):
                if is_episode_finished_batch[i]:
                    y_batch.append([reward_batch[i]])
                else:
                    y_batch.append(reward_batch[i] + GAMMA * q_value_batch[i])

            # Now that we have the y batch lets train the critic
            self.critic_network.train(y_batch, state_batch, action_batch)

            # Get the action batch so we can calculate the action gradient with it
            # Then get the action gradient batch and adapt the gradient with the gradient inverting method
            action_batch_for_gradients = self.actor_network.evaluate(
                state_batch)
            q_gradient_batch = self.critic_network.get_action_gradient(
                state_batch, action_batch_for_gradients)
            q_gradient_batch = self.grad_inv.invert(
                q_gradient_batch, action_batch_for_gradients)

            # Now we can train the actor
            self.actor_network.train(q_gradient_batch, state_batch, action_batch)

            # Save model if necessary
            if self.training_step > 0 and self.training_step % SAVE_STEP == 0:
                self.saver.save(self.session, NET_SAVE_PATH,
                                global_step=self.training_step)

            # Update time step
            self.training_step += 1

        self.data_manager.check_for_enqueue()

    def get_action(self, state):

        # normalize the state
        state = state.astype(float)
        state = np.divide(state, 100.0)

        # Get the action
        self.action = self.actor_network.get_action(state)

        # Are we using noise?
        if self.noise_flag:
            # scale noise down to 0 at training step 3000000
            if self.training_step < MAX_NOISE_STEP:
                self.action += (MAX_NOISE_STEP - self.training_step) / \
                    MAX_NOISE_STEP * self.exploration_noise.noise()
            # if action value lies outside of action bounds, rescale the action vector
            if self.action[0] < A0_BOUNDS[0] or self.action[0] > A0_BOUNDS[1]:
                self.action *= np.fabs(A0_BOUNDS[0] / self.action[0])
            if self.action[1] < A0_BOUNDS[0] or self.action[1] > A0_BOUNDS[1]:
                self.action *= np.fabs(A1_BOUNDS[0] / self.action[1])

        # Life q value output for this action and state
        self.print_q_value(state, self.action)

        return self.action

    def set_experience(self, state, reward, is_episode_finished):

        # Make sure we're saving a new old_state for the first experience of every episode
        if self.first_experience:
            self.first_experience = False
        else:
            self.data_manager.store_experience_to_file(self.old_state, self.old_action, reward, state,
                                                       is_episode_finished)

            # Uncomment if collecting data for the auto_encoder
            # experience = (self.old_state, self.old_action, reward, state, is_episode_finished)
            # self.buffer.append(experience)

        if is_episode_finished:
            self.first_experience = True
            self.exploration_noise.reset()

        # Safe old state and old action for next experience
        self.old_state = state
        self.old_action = self.action

    def print_q_value(self, state, action):

        string = "-"
        q_value = self.critic_network.evaluate([state], [action])
        stroke_pos = 30 * q_value[0][0] + 30
        if stroke_pos < 0:
            stroke_pos = 0
        elif stroke_pos > 60:
            stroke_pos = 60
        # print '[' + stroke_pos * string + '|' + (60-stroke_pos) * string + ']', "Q: ", q_value[0][0], \
            # "\tt: ", self.training_step
