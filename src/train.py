#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import os
import tensorflow as tf
from ddpg import DDPG
from ros_handler import RosHandler


dir_path = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL Follower Agent")
    parser.add_argument('--phase', default='train',
                        help='train or test')
    parser.add_argument('--gpu', default='0',
                        help='state the index of gpu: 0, 1, 2 or 3')
    parser.add_argument('--output_dir', default=dir_path+"/../output")
    # parser.add_argument('--learning_rate', default=0.001, type=float)
    # parser.add_argument('--max_steps', default=50000, type=int)
    # parser.add_argument('--pretrain', default='false',
    #                     help='true or false')
    args = parser.parse_args()
    return args


def main():

    agent = DDPG()

    print "========================== finish ddpg init?"
    rospy.init_node("robot_follower", anonymous=True)

    ros_handler = ROSHandler()

    print "============================ here??"
    while not rospy.is_shutdown():
        if ros_handler.new_msg():
            if not ros_handler.is_episode_finished:
                ros_handler.action = agent.get_action(ros_handler.state)

            agent.set_experience(ros_handler.state, ros_handler.reward,
                                 ros_handler.is_eipsode_finished)

            agent.learn()


if __name__ == '__main__':
    main()
