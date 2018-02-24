#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import numpy as np
from simulation_walk.msg import Laser4
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry


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

        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

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

    def _calculate_reward(self):


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
        rate = rospy.Rate(20)
        while not rospy.is_shutdown():
            self._pub.publish(self.action)
            rate.sleep()


if __name__=='__main__':
    ros_handler = RosHandler()
