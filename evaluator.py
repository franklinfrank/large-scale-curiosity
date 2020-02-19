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
        self.ep_len = 4500
        self.policy = policy
        if not os.path.exists('images'):
            os.mkdir('images')
        self.image_folder = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'images')
    
    def format_obs(self, obs_name, obs):
        nums = ",".join(map(str, obs))
        dict_format = "{" + nums + "}"
        final_str = "observation \"{}\" - {}\n".format(obs_name, dict_format)
        return final_str

    def eval_model(self, ep_num):
        for i in range(self.num_episodes):
            trajectory_file = self.exp_name + "_ep" + str(ep_num) + "_itr" + str(i) + "_trajectory.txt"
            if not os.path.exists("trajectories"):
                os.makedirs("trajectories")
            trajectory_path = os.path.join("trajectories", trajectory_file)
            ep_images = []
            ob = self.env.reset()
            ob = np.array(ob)
            eprews = []
            if i == 0:
                ep_images.append(self.env.unwrapped._last_observation)
            for step in range(self.ep_len):
                action, vpred, nlp = self.policy.get_ac_value_nlp_eval(ob)
                ob, rew, done, info = self.env.step(action[0])
                if i == 0:
                    ep_images.append(self.env.unwrapped._last_observation)
                if rew is None:
                    eprews.append(0)
                else:
                    eprews.append(rew)
                if step >  0:
                    pos_trans, pos_rot, vel_trans, vel_rot = self.env.unwrapped.get_pos_and_vel()

                    with open(trajectory_path, 'a') as f:
                        f.write(self.format_obs("DEBUG.POS.TRANS", pos_trans))
                        f.write(self.format_obs("DEBUG.POS.ROT", pos_rot))
                        f.write(self.format_obs("VEL.TRANS", vel_trans))
                        f.write(self.format_obs("VEL.ROT", vel_rot))
                
            for j in range(len(ep_images)):
                image_file = os.path.join(self.image_folder, self.exp_name +"_{}_{}_{}_".format(ep_num, i, j) + ".png")
                cv2.imwrite(image_file, ep_images[j])
            print("Episode {} cumulative reward: {}".format(i, sum(eprews)))
