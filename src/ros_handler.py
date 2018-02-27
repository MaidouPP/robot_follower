#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import rospy
import numpy as np
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
            "/reach_start", Point, self._gazebo_callback_new_start)
        self._sub_end_traj = rospy.Subscriber(
            "/end_traj", Bool, self._gazebo_callback_end_traj)

        self._pub_action = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._pub_robot_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self._pub_test = rospy.Publisher("/end_of_episode", Bool, queue_size=10)
        self._end_of_episode = Bool()
        self._end_of_episode.data = True

        # cost_map = rospy.get_param('costmap')
        # self._cost_map = pickle.load(costmap)

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
        msg = ModelState()
        msg.pose.position.x = self._person_pos[0]
        msg.pose.position.y = self._person_pos[1]
        msg.model_name = "freight"
        # data: Point message
        x, y = _calculate_start_pos(data)
        self._pub_robot_pos.publish(msg)

    def _calculate_start_pos(self, target, actor_pos):
        dx = target.x - actor_pos[0]
        dy = target.y - actor_pos[1]
        radius = random.uniform(0.3, 1)


    def _calculate_reward(self):
        pass

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
