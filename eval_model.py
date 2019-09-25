import tensorflow as tf
import gym
import gym_deepmindlab
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
    def __init__(self, env, num_episodes, exp_name, test_env, args):
        self.exp_name = exp_name
        self.env = env
        self.num_episodes = num_episodes
        self.hps=args
#        self.ob_mean, self.ob_std = random_agent_ob_mean_std(test_env)
        del test_env
    def eval_model(self):
        hps = self.hps
        tf.reset_default_graph()
#        imported_graph = tf.train.import_meta_graph("/tmp/" + self.exp_name + ".ckpt.meta")
        with tf.Session() as sess:
            self.policy = self.policy = CnnPolicy(
                scope='pol',
                ob_space=self.env.observation_space,
                ac_space=self.env.action_space,
                hidsize=512,
                feat_dim=512,
                ob_mean=0,
                ob_std=0,
                layernormalize=False,
                nl=tf.nn.leaky_relu)
#            self.feature_extractor = {"none": FeatureExtractor,
#                                  "idf": InverseDynamics,
#                                  "vaesph": partial(VAE, spherical_obs=True),
#                                  "vaenonsph": partial(VAE, spherical_obs=False),
#                                  "pix2pix": JustPixels}[hps['feat_learning']]
#            self.feature_extractor = self.feature_extractor(policy=self.policy,
#                                                        features_shared_with_policy=False,
#                                                        feat_dim=512,
#                                                        layernormalize=hps['layernorm'])
#
#            self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else Unet
#            self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
#                                      predict_from_pixels=hps['dyn_from_pixels'],
#                                      feat_dim=512)
#
#            self.agent = PpoOptimizer(
#                scope='ppo',
#                ob_space=self.env.observation_space,
#                ac_space=self.env.action_space,
#                stochpol=self.policy,
#                use_news=hps['use_news'],
#                gamma=hps['gamma'],
#                lam=hps["lambda"],
#                nepochs=hps['nepochs'],
#                nminibatches=hps['nminibatches'],
#                lr=hps['lr'],
#                cliprange=0.1,
#                nsteps_per_seg=hps['nsteps_per_seg'],
#                nsegs_per_env=hps['nsegs_per_env'],
#                ent_coef=hps['ent_coeff'],
#                normrew=hps['norm_rew'],
#                normadv=hps['norm_adv'],
#                ext_coeff=hps['ext_coeff'],
#                int_coeff=hps['int_coeff'],
#                dynamics=self.dynamics,
#                exp_name = hps['exp_name']
#            )
#            self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
#            self.agent.total_loss += self.agent.to_report['aux']
#            self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
#            self.agent.total_loss += self.agent.to_report['dyn_loss']
#            self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])
            self.policy.restore_model(self.exp_name)
#            imported_graph.restore(sess,"/tmp/" + self.exp_name + ".ckpt")
#            chkp.print_tensors_in_checkpoint_file("/tmp/" + self.exp_name +".ckpt", tensor_name = '', all_tensors=True)
            print("Model restored")
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
                        image_file = os.path.join(dirname, image_folder + "/" + self.exp_name + "000" + str(i) + ".png")
                        cv2.imwrite(image_file, ep_images[i])
                        print("Image written to {}".format(image_file))
                print("Episode {} cumulative reward: {}".format(i, sum(eprews)))




def start_eval(**args):
    print("Starting model evaluation")
    env = gym.make(args['env'])
    test_env = gym.make(args['env'])
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    test_env = ProcessFrame84(test_env, crop=False)
    test_env = FrameStack(test_env, 4)
    num_episodes = args['num_episodes']
    exp_name = args['exp_name']
    evaluator = Evaluator(env, num_episodes, exp_name, test_env, args)
    evaluator.eval_model()


def add_optimization_params(parser):
    parser.add_argument('--lambda', type=float, default=0.95)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--nminibatches', type=int, default=8)
    parser.add_argument('--norm_adv', type=int, default=1)
    parser.add_argument('--norm_rew', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--ent_coeff', type=float, default=0.001)
    parser.add_argument('--nepochs', type=int, default=3)
    parser.add_argument('--num_timesteps', type=int, default=int(1e8))


def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=900)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_optimization_params(parser)
    add_rollout_params(parser)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--expID', type=str, default='000')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--dyn_from_pixels', type=int, default=0)
    parser.add_argument('--use_news', type=int, default=0)
    parser.add_argument('--ext_coeff', type=float, default=0.)
    parser.add_argument('--int_coeff', type=float, default=1.)
    parser.add_argument('--layernorm', type=int, default=0)
    parser.add_argument('--feat_learning', type=str, default="none",
                        choices=["none", "idf", "vaesph", "vaenonsph", "pix2pix"])
    parser.add_argument('--env', type=str, default='DeepmindLabNavMazeStatic01-v0')
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    start_eval(**args.__dict__)


