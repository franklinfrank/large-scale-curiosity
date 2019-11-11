#!/usr/bin/env python
try:
    from OpenGL import GLU
except:
    print("no OpenGL.GLU")
import functools
import os.path as osp
from functools import partial

import gym
import gym_deepmindlab
import datetime
import tensorflow as tf
from baselines import logger
from baselines.bench import Monitor
from baselines.common.atari_wrappers import NoopResetEnv, FrameStack
from mpi4py import MPI

from auxiliary_tasks import FeatureExtractor, InverseDynamics, VAE, JustPixels
from cnn_policy import CnnPolicy
from cppo_agent import PpoOptimizer
from dynamics import Dynamics, UNet
from utils import random_agent_ob_mean_std
from wrappers import MontezumaInfoWrapper, make_mario_env, make_robo_pong, make_robo_hockey, \
    make_multi_pong, AddRandomStateToInfo, MaxAndSkipEnv, ProcessFrame84, ExtraTimeLimit


def start_experiment(**args):
    make_env = partial(make_env_all_params, add_monitor=True, args=args)
    trainer = Trainer(make_env=make_env,
                      num_timesteps=args['num_timesteps'], hps=args,
                      envs_per_process=args['envs_per_process'])
    log, tf_sess = get_experiment_environment(**args)

    with log, tf_sess:
        logdir = logger.get_dir()
        print("results will be saved to ", logdir)
        trainer.train()
        policy = trainer.policy
        feat_ext = trainer.feature_extractor
        dyn = trainer.dynamics
    if args['tune_envs']:
        for i in range(len(args['tune_envs'])):
            
            sess = tf.Session() 
            tf.reset_default_graph
            with log, sess:
                tune_make_env = partial(make_tune_env, add_monitor=True, args=args, tune_num=i)
                new_exp = args['exp_name'] + "_tune_on_{}".format(args['tune_envs'][i])
                new_trainer = Trainer(make_env=tune_make_env, num_timesteps=args['num_timesteps_tune'],
                                      hps=args, envs_per_process=args['envs_per_process'], exp_name=new_exp,
                                      env_name=args['tune_envs'][i], policy=policy,
                                      feat_ext=feat_ext, dyn=dyn, agent_num=i)
                new_trainer.agent.restore_model(args['exp_name'] + "_final")
                new_trainer.train()


class Trainer(object):
    def __init__(self, make_env, hps, num_timesteps, envs_per_process, exp_name=None, env_name=None, policy=None, feat_ext=None, dyn=None, agent_num=None):
        self.make_env = make_env
        self.hps = hps
        self.envs_per_process = envs_per_process
        self.num_timesteps = num_timesteps
        self._set_env_vars()
        if exp_name:
            self.exp_name = exp_name
        else:
            self.exp_name = hps['exp_name']
        if env_name:
            self.env_name = env_name
        else:
            self.env_name = hps['env']

        if policy is None:
            self.policy = CnnPolicy(
                scope='pol',
                ob_space=self.ob_space,
                ac_space=self.ac_space,
                hidsize=512,
                feat_dim=512,
                ob_mean=self.ob_mean,
                ob_std=self.ob_std,
                layernormalize=False,
                nl=tf.nn.leaky_relu)
        else:
            self.policy = policy
        if exp_name:
            self.policy.restore_model(hps['exp_name'] +  "_final")

        if feat_ext:
            self.feature_extractor = feat_ext
        else:

            self.feature_extractor = {"none": FeatureExtractor,
                                      "idf": InverseDynamics,
                                      "vaesph": partial(VAE, spherical_obs=True),
                                      "vaenonsph": partial(VAE, spherical_obs=False),
                                      "pix2pix": JustPixels}[hps['feat_learning']]

            self.feature_extractor = self.feature_extractor(policy=self.policy,
                                                            features_shared_with_policy=False,
                                                            feat_dim=512,
                                                            layernormalize=hps['layernorm'])
        if dyn:
            self.dynamics = dyn
        else:

            self.dynamics = Dynamics if hps['feat_learning'] != 'pix2pix' else UNet
            self.dynamics = self.dynamics(auxiliary_task=self.feature_extractor,
                                          predict_from_pixels=hps['dyn_from_pixels'],
                                          feat_dim=512)
        self.agent = PpoOptimizer(
            scope='ppo',
            ob_space=self.ob_space,
            ac_space=self.ac_space,
            stochpol=self.policy,
            use_news=hps['use_news'],
            gamma=hps['gamma'],
            lam=hps["lambda"],
            nepochs=hps['nepochs'],
            nminibatches=hps['nminibatches'],
            lr=hps['lr'],
            cliprange=0.1,
            nsteps_per_seg=hps['nsteps_per_seg'],
            nsegs_per_env=hps['nsegs_per_env'],
            ent_coef=hps['ent_coeff'],
            normrew=hps['norm_rew'],
            normadv=hps['norm_adv'],
            ext_coeff=hps['ext_coeff'],
            int_coeff=hps['int_coeff'],
            dynamics=self.dynamics,
            exp_name=self.exp_name,
            env_name=self.env_name,
            video_log_freq=hps['video_log_freq'],
            model_save_freq=hps['model_save_freq'],
            use_apples=hps['use_apples'],
            agent_num=agent_num
        )

        self.agent.to_report['aux'] = tf.reduce_mean(self.feature_extractor.loss)
        self.agent.total_loss += self.agent.to_report['aux']
        self.agent.to_report['dyn_loss'] = tf.reduce_mean(self.dynamics.loss)
        self.agent.total_loss += self.agent.to_report['dyn_loss']
        self.agent.to_report['feat_var'] = tf.reduce_mean(tf.nn.moments(self.feature_extractor.features, [0, 1])[1])

    def _set_env_vars(self):
        env = self.make_env(0, add_monitor=False)
        self.ob_space, self.ac_space = env.observation_space, env.action_space
        self.ob_mean, self.ob_std = random_agent_ob_mean_std(env)
        del env
        self.envs = [functools.partial(self.make_env, i) for i in range(self.envs_per_process)]

    def train(self):
        self.agent.start_interaction(self.envs, nlump=self.hps['nlumps'], dynamics=self.dynamics)
        while True:
            info = self.agent.step()
            if info['update']:
                logger.logkvs(info['update'])
                logger.dumpkvs()
            if self.agent.rollout.stats['tcount'] > self.num_timesteps:
                break

        self.agent.stop_interaction()
        self.policy.save_model(self.exp_name, None)



def make_env_all_params(rank, add_monitor, args):
    if args["env_kind"] == 'deepmind':
        env = gym.make(args['env'])
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
    elif args["env_kind"] == 'atari':
        env = gym.make(args['env'])
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=args['noop_max'])
        env = MaxAndSkipEnv(env, skip=4)
        env = ProcessFrame84(env, crop=False)
        env = FrameStack(env, 4)
        env = ExtraTimeLimit(env, args['max_episode_steps'])
        if 'Montezuma' in args['env']:
            env = MontezumaInfoWrapper(env)
        env = AddRandomStateToInfo(env)
    elif args["env_kind"] == 'mario':
        env = make_mario_env()
    elif args["env_kind"] == "retro_multi":
        env = make_multi_pong()
    elif args["env_kind"] == 'robopong':
        if args["env"] == "pong":
            env = make_robo_pong()
        elif args["env"] == "hockey":
            env = make_robo_hockey()

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env

def make_tune_env(rank, add_monitor, args, tune_num):

    env = gym.make(args['tune_envs'][tune_num])
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)

    if add_monitor:
        env = Monitor(env, osp.join(logger.get_dir(), '%.2i' % rank))
    return env


def get_experiment_environment(**args):
    from utils import setup_mpi_gpus, setup_tensorflow_session
    from baselines.common import set_global_seeds
    from gym.utils.seeding import hash_seed
    process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
    process_seed = hash_seed(process_seed, max_bytes=4)
    set_global_seeds(process_seed)
    setup_mpi_gpus()

    logger_context = logger.scoped_configure(dir='./logs/' + 
                        datetime.datetime.now().strftime(args["expID"] + "-openai-%Y-%m-%d-%H-%M-%S-%f"),
                         format_strs=['stdout', 'log', 'csv', 'tensorboard'] if MPI.COMM_WORLD.Get_rank() == 0 else ['log'])
    tf_context = setup_tensorflow_session()
    return logger_context, tf_context


def add_environments_params(parser):
    parser.add_argument('--env', help='environment ID', default='DeepmindLabNavMazeStatic01-v0',
                        type=str)
    parser.add_argument('--max-episode-steps', help='maximum number of timesteps for episode', default=7200, type=int)
    parser.add_argument('--env_kind', type=str, default="deepmind")
    parser.add_argument('--noop_max', type=int, default=30)
    parser.add_argument('--tune_envs', type=str, nargs='+', default=None)

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
    parser.add_argument('--num_timesteps_tune', type=int, default=int(1e7))

def add_rollout_params(parser):
    parser.add_argument('--nsteps_per_seg', type=int, default=900)
    parser.add_argument('--nsegs_per_env', type=int, default=1)
    parser.add_argument('--envs_per_process', type=int, default=128)
    parser.add_argument('--nlumps', type=int, default=1)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_environments_params(parser)
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
    parser.add_argument('--video_log_freq', type=int, default=100)
    parser.add_argument('--model_save_freq', type=int, default=25)
    parser.add_argument('--use_apples', type=int, default=1)

    args = parser.parse_args()

    start_experiment(**args.__dict__)
