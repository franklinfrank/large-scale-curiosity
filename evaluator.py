from wrappers import ProcessFrame84
from baselines.common.atari_wrappers import FrameStack
from baselines.common.distributions import make_pdtype
from utils import setup_tensorflow_session
from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim, random_agent_ob_mean_std
from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
import numpy as np
from functools import partial
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
import cv2
import os
from tensorflow.python.tools import inspect_checkpoint as chkp

class Evaluator(object):
    def __init__(self, env_name, num_episodes, exp_name):
        self.exp_name = exp_name
        self.env = gym.make(env_name)
        self.num_episodes = num_episodes
        self.policy = policy

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
            if i == 0:
                dirname = os.path.abspath(os.path.dirname(__file__))
                image_folder = 'images'
                for i in range(len(ep_images)):
                    image_file = os.path.join(dirname, image_folder + "/" + self.exp_name + +"_{}_{}_".format(ep_num, i) + ".png")
                    cv2.imwrite(image_file, ep_images[i])
                    print("Image written to {}".format(image_file))
            print("Episode {} cumulative reward: {}".format(i, sum(eprews)))