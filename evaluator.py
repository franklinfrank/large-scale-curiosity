from wrappers import ProcessFrame84
from baselines.common.atari_wrappers import FrameStack
from baselines.common.distributions import make_pdtype
import numpy as np
from cnn_policy import CnnPolicy
import cv2
import os
from tensorflow.python.tools import inspect_checkpoint as chkp
import gym
import gym_deepmindlab

class Evaluator(object):
    def __init__(self, env_name, num_episodes, exp_name, policy):
        self.exp_name = exp_name
        self.env = gym.make(env_name)
        self.env = ProcessFrame84(self.env, crop=False)
        self.env = FrameStack(self.env, 4)
        self.num_episodes = 1
        self.policy = policy
	if not path.exists('images'):
	    os.mkdir('images')
        self.image_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'images')

    def eval_model(self, ep_num):
        for i in range(self.num_episodes):
            ep_images = []
            ob = self.env.reset()
            ob = np.array(ob)
            eprews = []
            if i == 0:
                ep_images.append(self.env.unwrapped._last_observation)
            for step in range(900):
                action, vpred, nlp = self.policy.get_ac_value_nlp_eval(ob)
                ob, rew, done, info = self.env.step(action[0])
                if i == 0:
                    ep_images.append(self.env.unwrapped._last_observation)
                if rew is None:
                    eprews.append(0)
                else:
                    eprews.append(rew)
                if done:
                    print("Episode finished after {} timesteps".format(step+1))
                    print("Episode Reward is {}".format(sum(eprews)))
                    break
            dirname = os.path.abspath(os.path.dirname(__file__))
            image_folder = 'images'
            for j in range(len(ep_images)):
                image_file = os.path.join(self.image_folder, self.exp_name +"_{}_{}_{}_".format(ep_num, i, j) + ".png")
                cv2.imwrite(image_file, ep_images[j])
            print("Episode {} cumulative reward: {}".format(i, sum(eprews)))
