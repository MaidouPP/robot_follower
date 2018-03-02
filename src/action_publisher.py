#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool


class ActionPublisher:

    def __init__(self):
        self._action = Twist()
        self._sub_new_act = rospy.Subscriber(
            "/cmd_vel_handler", Twist, self._cmd_vel_receiver)
        self._sub_new_start = rospy.Subscriber(
            "/new_start", Point, self._gazebo_callback_new_start)
        self._sub_end_traj = rospy.Subscriber(
            "/reach_dest", Bool, self._gazebo_callback_end_traj)
        self._sub_end_traj = rospy.Subscriber(
            "/bump", Bool, self._gazebo_callback_bump)

        self._pub_new_act = rospy.Publisher(
            "/cmd_vel", Twist, queue_size=10)

        self._ready = False
        rospy.init_node('action_publisher', anonymous=True)

        self._publish_action()

    def _gazebo_callback_end_traj(self, data):
        self._ready = False
        self._action = Twist()

    def _gazebo_callback_new_start(self, data):
        self._ready = True

    def _gazebo_callback_bump(self, data):
        self._ready = False
        self._action = Twist()

    def _cmd_vel_receiver(self, data):
        self._action = data

    def _publish_action(self):
        rate = rospy.Rate(0.02)
        while not rospy.is_shutdown():
            if self._ready:
                self._pub_new_act.publish(self._action)
                # self._pub_test.publish(self._end_of_episode)
                rate.sleep()

if __name__ == "__main__":

    act_pub = ActionPublisher()
