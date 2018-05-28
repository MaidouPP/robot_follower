#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle as pickle
import rospy
import math
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

        self._depth = 1
        self._length = 2  # SICK TIM561 laser scanner dimension
        # self._state = np.zeros((1, self._length, self._depth), dtype='float32')
        self._state = np.zeros((1, self._length))

        self._reward = 0.0
        self._action = Twist()
        self._old_action = Twist()
        self._action.linear.x = self._action.linear.y = \
            self._action.linear.z = 0
        self._action.angular.x = self._action.angular.y = \
            self._action.angular.z = 0
        # self._action.linear.x = 1
        # self._action.angular.z = 0.5
        self._person_pos = np.zeros((2,))
        self._person_target = np.zeros((2,))
        self._robot_pos = np.zeros((2,))

        cost_map = rospy.get_param('costmap')
        pickle_file = open(cost_map, 'rb')
        self._cost_map = pickle.load(pickle_file)
        rospy.sleep(0.2)

        self._sub_laser = rospy.Subscriber(
            "/simulation_walk/laser4", Laser4, self._input_callback_laser)
        self._sub_person_pos = rospy.Subscriber(
            "/actor_pos", Point, self._input_callback_person_pos)
        self._sub_person_target = rospy.Subscriber(
            "/actor_target", Point, self._input_callback_person_target)
        self._sub_robot_pos = rospy.Subscriber(
            "/ground_truth/state", Odometry, self._input_callback_robot_pos)
        self._sub_new_start = rospy.Subscriber(
            "/new_start", Point, self._gazebo_callback_new_start)
        self._sub_end_traj = rospy.Subscriber(
            "/reach_dest", Bool, self._gazebo_callback_end_traj)

        self._pub_action = rospy.Publisher("/cmd_vel_handler", Twist, queue_size=10)
        self._pub_robot_pos = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self._pub_end = rospy.Publisher("/bump", Bool, queue_size=10)
        self.end_of_episode = False
        # self._end_of_episode.data = True

        # rospy.init_node("ros_handler", anonymous=True)

        self._new_msg_flag = False

        # self._publish_action()

    def _input_callback_laser(self, data):
        # ranges = np.array(data.ranges)
        # print ranges.shape
        # self._state = ranges.reshape((1, self._length, self._depth))
        # self._state[self._state == np.inf] = 30
        # self._state = self._state / 10.0
        self._state = self._person_pos - self._robot_pos
        # self._state = self._state.reshape((1, -1))
        self._reward = self._calculate_reward()
        self._new_msg_flag = True

    def _input_callback_person_pos(self, data):
        self._person_pos[0] = data.x
        self._person_pos[1] = data.y

    def _input_callback_person_target(self, data):
        self._person_target[0] = data.x
        self._person_target[1] = data.y
        # print self._person_target

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
        self.end_of_episode = False
        rospy.sleep(0.2)
        self._calculate_start_pos(data, self._person_pos)

    def _gazebo_callback_end_traj(self, data):
        self.end_of_episode = True

    def _calculate_start_pos(self, target, actor_pos):
        """
        calculate the starting point for robot to follow
        the person in the new round
        """
        dx = target.x - actor_pos[0]
        dy = target.y - actor_pos[1]
        vec = np.array([dx, dy])
        # norm = np.linalg.norm(vec)
        vec = self._normalize_vec(vec)
        # vec = vec / norm
        ortho = np.array([vec[1], -vec[0]])
        success = False
        result = np.zeros((2,))

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
        pos_ = np.array([pos[0], pos[1]])
        if pos[0] > 10 or pos[0] < -10:
            return False
        elif pos[1] > 10 or pos[1] < -10:
            return False
        # may need to modify this part because
        # it relates to what map resolution we are using
        # and what map size...
        # for now it limits to 2000*2000 map with real 20*20
        # pos_[0] = pos_[0] + 10
        # pos_[1] = -pos_[1] + 10
        # pos_ = pos_ * 100
        # # print pos_, " ", pos
        # # print self._cost_map[int(pos_[1]), int(pos_[0])]
        vec = self._person_pos - self._robot_pos
        distance = np.linalg.norm(vec)
        # # print "=== pos_: ", pos_
        # # print "--- pos: ", pos
        if distance < 0.3:
            return False
        # if self._cost_map[int(pos_[1]), int(pos_[0])] == 0:
        #     return True
        # else:
        #     return False
        return True

    def _calculate_reward(self):
        v1 = self._person_target - self._person_pos
        v2 = self._person_pos - self._robot_pos
        v1_ = self._normalize_vec(v1)
        v2_ = self._normalize_vec(v2)
        ortho_v1 = np.array([v1_[1], -v1_[0]])
        # act_punish = -10 if math.fabs(self.action[0][0]) > 0.8 \
        #                     or math.fabs(self.action[0][1]) > 0.8 \
        #               else 0

        reward = 0
        # reward += act_punish

        if not self._valid_pos(self._robot_pos):
            self.end_of_episode = True
            msg = Bool()
            msg.data = True
            self._pub_end.publish(msg)

            print "Died!!! reward is: ", reward - 20
            return reward - 20.0

        if np.dot(v1_, v2_) < 0:
            reward -= 1.0
        else:
            distance = np.linalg.norm(v2)
            if distance > 5:
                reward -= 3.0
            elif distance > 3:
                reward -= 2.0
            elif distance > 2:
                reward -= 1.0
            else:
                if (self._calculate_angle(ortho_v1, v2_) < math.pi/4 and \
                   self._calculate_angle(ortho_v1, v2_) > 0) or \
                   (self._calculate_angle(-ortho_v1, v2_) < math.pi/4 and \
                    self._calculate_angle(-ortho_v1, v2_) > 0):
                    reward = reward
                else:
                    reward -= 0.1
        print "      reward: ", reward
        return reward

    @staticmethod
    def _calculate_angle(vec1, vec2):
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 and norm2:
            return math.acos(np.dot(vec1, vec2) / (norm1*norm2))
        else:
            return 0

    @staticmethod
    def _normalize_vec(vec):
        norm = np.linalg.norm(vec)
        vec = vec / norm
        return vec

    @staticmethod
    def _yaw_to_quaternion(vec):
        z = np.arctan2(vec[1], vec[0])
        quaternion = tf.transformations.quaternion_from_euler(0, 0, z)
        return quaternion

    @property
    def action(self):
        tmp = np.array([self._action.linear.x,
                         self._action.angular.z])
        tmp = np.reshape(tmp, (1, 2))
        return tmp

    @action.setter
    def action(self, vel):
        self._action.linear.x, self._action.angular.z = vel

    @property
    def old_action(self):
        tmp = np.array([self._old_action.linear.x,
                         self._old_action.angular.z])
        tmp = np.reshape(tmp, (1, 2))
        return tmp

    @property
    def state(self):
        return self._state

    @property
    def reward(self):
        return self._reward

    def new_msg(self):
        """
        return true if new message arrives and set new_msg_flag to False later
        """
        output = False
        if self._new_msg_flag:
            output = True
            self._new_msg_flag = False

        return output

    def publish_action(self, new_action):
        self._old_action = self._action
        self._action.linear.x, self._action.angular.z = new_action
        self._pub_action.publish(self._action)


if __name__=='__main__':
    ros_handler = RosHandler()
