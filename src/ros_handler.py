#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import rospy
import numpy as np
import random
import tf
from gazebo_msgs.msg import ModelState
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from simulation_walk.msg import Laser4
from std_msgs.msg import Bool


class RosHandler:

    def __init__(self):

        self._init = False

        self._depth = 4
        self._length = 662   # SICK TIM561 laser scanner dimension
        self._state = np.zeros((self._length, self._depth), dtype='float32')

        self._reward = 0.0
        self._action = Twist()
        self._action.linear.x = self._action.linear.y = \
            self._action.linear.z = 0
        self._action.angular.x = self._action.angular.y = \
            self._action.angular.z = 0
        self._action.linear.x = 1
        self._action.angular.z = 0.5
        self._person_pos = np.zeros((2,))
        self._robot_pos = np.zeros((2,))

        self._sub_laser = rospy.Subscriber(
            "/simulation_walk/laser4", Laser4, self._input_callback_laser)
        self._sub_person_pos = rospy.Subscriber(
            "/actor_pos", Point, self._input_callback_person_pos)
        self._sub_robot_pos = rospy.Subscriber(
            "/ground_truth/state", Odometry, self._input_callback_robot_pos)
        self._sub_new_start = rospy.Subscriber(
            "/new_start", Point, self._gazebo_callback_new_start)
        self._sub_end_traj = rospy.Subscriber(
            "/end_traj", Bool, self._gazebo_callback_end_traj)

        self._pub_action = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._pub_robot_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self._pub_test = rospy.Publisher("/end_of_episode", Bool, queue_size=10)
        self._end_of_episode = Bool()
        self._end_of_episode.data = True

        cost_map = rospy.get_param('costmap')
        with open(cost_map, 'rb') as pickle_file:
            self._cost_map = pickle.load(pickle_file)
        # self._cost_map = pickle.load(cost_map)

        rospy.init_node("ros_handler", anonymous=True)

        self._new_msg_flag = False

        self._publish_action()

    def _input_callback_laser(self, data):
        ranges = np.array(data.ranges)
        print ranges.shape
        self._state = ranges.reshape((self._length, self._depth))
        self._calculate_reward()
        self._new_msg_flag = True

    def _input_callback_person_pos(self, data):
        self._person_pos[0] = data.x
        self._person_pos[1] = data.y

    def _input_callback_robot_pos(self, data):
        self._robot_pos[0] = data.pose.pose.position.x
        self._robot_pos[1] = data.pose.pose.position.y

    def _gazebo_callback_new_start(self, data):
        # msg = ModelState()
        # msg.pose.position.x = self._person_pos[0]
        # msg.pose.position.y = self._person_pos[1]
        # msg.model_name = "freight"
        # data: Point message
        # print "ever here?? 86"
        # print "=============== the target is: ", data
        # print "=============== the human isL ", self._person_pos
        self._calculate_start_pos(data, self._person_pos)

    def _gazebo_callback_end_traj(self, data):
        pass

    def _calculate_start_pos(self, target, actor_pos):
        dx = target.x - actor_pos[0]
        dy = target.y - actor_pos[1]
        vec = np.array([dx, dy])
        norm = np.linalg.norm(vec)
        vec = vec / norm
        ortho = np.array([vec[1], -vec[0]])
        success = False
        result = np.zeros((2, 1))

        while success is False:

            delta_r = random.uniform(0.5, 1)
            temp = np.array([actor_pos[0]-delta_r*vec[0],
                             actor_pos[1]-delta_r*vec[1]])
            noise = random.uniform(-0.2, 0.2)
            result[0] = temp[0] + noise * ortho[0]
            result[1] = temp[1] + noise * ortho[1]

            success = self._valid_pos(result)

        msg = ModelState()
        msg.model_name = "freight"
        msg.pose.position.x, msg.pose.position.y = result[0], result[1]
        quaternion = self._yaw_to_quaternion(vec)
        msg.pose.orientation.x = quaternion[0]
        msg.pose.orientation.y = quaternion[1]
        msg.pose.orientation.z = quaternion[2]
        msg.pose.orientation.w = quaternion[3]
        self._pub_robot_pos.publish(msg)

    def _valid_pos(self, pos):
        """
        check if this place is in safe zone for robot
        """
        pos_ = [pos[0], pos[1]]
        pos_ = np.array(pos_ + [10, 10]) * 100
        if self._cost_map[int(pos_[0]), int(pos_[1])] == 0:
            return True
        else:
            return False

    def _calculate_reward(self):
        pass

    @staticmethod
    def _yaw_to_quaternion(vec):
        z = np.arctan2(vec[1], vec[0])
        quaternion = tf.transformations.quaternion_from_euler(0, 0, z)
        return quaternion

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, vel):
        try:
            linear_x, angular_z = vel
        except ValueError:
            raise ValueError("Pass an iterable with two elements")
        else:
            self._action.linear_x = linear_x
            self._action.angular_z = angular_z

    @property
    def state(self):
        return self._state

    def new_msg(self):
        """
        return true if new message arrives and set new_msg_flag to False later
        """
        output = False
        if self._new_msg_flag:
            output = True
            self._new_msg_flag = False

        return output

    def _publish_action(self):
        rate = rospy.Rate(0.1)
        while not rospy.is_shutdown():
            self._pub_action.publish(self.action)
            # self._pub_test.publish(self._end_of_episode)
            rate.sleep()


if __name__=='__main__':
    ros_handler = RosHandler()
