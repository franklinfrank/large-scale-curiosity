import time

import numpy as np
import tensorflow.compat.v1 as tf
from baselines.common import explained_variance
from baselines.common.mpi_moments import mpi_moments
from baselines.common.running_mean_std import RunningMeanStd
from mpi4py import MPI

from mpi_utils import MpiAdamOptimizer
from rollouts import Rollout
from utils import bcast_tf_vars_from_root, get_mean_and_std
from vec_env import ShmemVecEnv as VecEnv
from evaluator import Evaluator
tf.disable_v2_behavior()
getsess = tf.get_default_session


class PpoOptimizer(object):
    envs = None

    def __init__(self, *, scope, ob_space, ac_space, stochpol,
                 ent_coef, gamma, lam, nepochs, lr, cliprange,
                 nminibatches,
                 normrew, normadv, use_news, ext_coeff, int_coeff,
                 nsteps_per_seg, nsegs_per_env, dynamics, exp_name, env_name, video_log_freq, model_save_freq,
                 use_apples, agent_num=None, restore_name=None, multi_envs=None, lstm=False, lstm1_size=512, lstm2_size=0):
        self.dynamics = dynamics
        self.exp_name = exp_name
        self.env_name = env_name
        self.video_log_freq = video_log_freq
        self.model_save_freq = model_save_freq
        self.use_apples = use_apples
        self.agent_num = agent_num
        self.multi_envs = multi_envs
        self.lstm = lstm
        self.lstm1_size = lstm1_size
        self.lstm2_size = lstm2_size
        with tf.variable_scope(scope):
            self.use_recorder = True
            self.n_updates = 0
            self.scope = scope
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.stochpol = stochpol
            self.nepochs = nepochs
            self.lr = lr
            self.cliprange = cliprange
            self.nsteps_per_seg = nsteps_per_seg
            self.nsegs_per_env = nsegs_per_env
            self.nminibatches = nminibatches
            self.gamma = gamma
            self.lam = lam
            self.normrew = normrew
            self.normadv = normadv
            self.use_news = use_news
            self.ext_coeff = ext_coeff
            self.int_coeff = int_coeff
            self.ent_coeff = ent_coef
            if self.agent_num is None:
                self.ph_adv = tf.placeholder(tf.float32, [None, None], name='adv')
                self.ph_ret = tf.placeholder(tf.float32, [None, None], name='ret')
                self.ph_rews = tf.placeholder(tf.float32, [None, None], name='rews')
                self.ph_oldnlp = tf.placeholder(tf.float32, [None, None], name='oldnlp')
                self.ph_oldvpred = tf.placeholder(tf.float32, [None, None], name='oldvpred')
                self.ph_lr = tf.placeholder(tf.float32, [], name='lr')
                self.ph_cliprange = tf.placeholder(tf.float32, [], name='cliprange')
                neglogpac = self.stochpol.pd.neglogp(self.stochpol.ph_ac)
                entropy = tf.reduce_mean(self.stochpol.pd.entropy(), name='agent_entropy')
                vpred = self.stochpol.vpred
                vf_loss = 0.5 * tf.reduce_mean((vpred - self.ph_ret) ** 2, name='vf_loss')
                ratio = tf.exp(self.ph_oldnlp - neglogpac, name='ratio')  # p_new / p_old
                negadv = - self.ph_adv
                pg_losses1 = negadv * ratio
                pg_losses2 = negadv * tf.clip_by_value(ratio, 1.0 - self.ph_cliprange, 1.0 + self.ph_cliprange, name='pglosses2')
                pg_loss_surr = tf.maximum(pg_losses1, pg_losses2, name='loss_surr')
                pg_loss = tf.reduce_mean(pg_loss_surr, name='pg_loss')
                ent_loss = (- ent_coef) * entropy
                approxkl = .5 * tf.reduce_mean(tf.square(neglogpac - self.ph_oldnlp), name='approxkl')
                clipfrac = tf.reduce_mean(tf.to_float(tf.abs(pg_losses2 - pg_loss_surr) > 1e-6), name='clipfrac')

                self.total_loss = pg_loss + ent_loss + vf_loss
                self.to_report = {'tot': self.total_loss, 'pg': pg_loss, 'vf': vf_loss, 'ent': entropy,
                              'approxkl': approxkl, 'clipfrac': clipfrac}
                tf.add_to_collection('adv', self.ph_adv)
                tf.add_to_collection('ret', self.ph_ret)
                tf.add_to_collection('rews', self.ph_rews)
                tf.add_to_collection('oldnlp', self.ph_oldnlp)
                tf.add_to_collection('oldvpred', self.ph_oldvpred)
                tf.add_to_collection('lr', self.ph_lr)
                tf.add_to_collection('cliprange', self.ph_cliprange)
                tf.add_to_collection('agent_entropy', entropy)
                tf.add_to_collection('vf_loss', vf_loss)
                tf.add_to_collection('ratio', ratio)
                tf.add_to_collection('pg_losses2', pg_losses2)
                tf.add_to_collection('loss_surr', pg_loss_surr)
                tf.add_to_collection('pg_loss', pg_loss)
                tf.add_to_collection('approxkl', approxkl)
                tf.add_to_collection('clipfrac', clipfrac)
            else:
                self.restore()

    def start_interaction(self, env_fns, dynamics, nlump=2):
        self.loss_names, self._losses = zip(*list(self.to_report.items()))

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if MPI.COMM_WORLD.Get_size() > 1:
            if self.agent_num is None:
                trainer = MpiAdamOptimizer(learning_rate=self.ph_lr, comm=MPI.COMM_WORLD)
                
        else:
            if self.agent_num is None:
                trainer = tf.train.AdamOptimizer(learning_rate=self.ph_lr)
        if self.agent_num is None:
            gradsandvars = trainer.compute_gradients(self.total_loss, params)
        
            self._train = trainer.apply_gradients(gradsandvars)
            tf.add_to_collection("train_op", self._train)
        else:
            self._train = tf.get_collection("train_op")[0]

        if MPI.COMM_WORLD.Get_rank() == 0:
            getsess().run(tf.variables_initializer(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)))
        bcast_tf_vars_from_root(getsess(), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

        self.all_visited_rooms = []
        self.all_scores = []
        self.nenvs = nenvs = len(env_fns)
        self.nlump = nlump
        self.lump_stride = nenvs // self.nlump
        self.envs = [
            VecEnv(env_fns[l * self.lump_stride: (l + 1) * self.lump_stride], spaces=[self.ob_space, self.ac_space]) for
            l in range(self.nlump)]

        self.rollout = Rollout(ob_space=self.ob_space, ac_space=self.ac_space, nenvs=nenvs,
                               nsteps_per_seg=self.nsteps_per_seg,
                               nsegs_per_env=self.nsegs_per_env, nlumps=self.nlump,
                               envs=self.envs,
                               policy=self.stochpol,
                               int_rew_coeff=self.int_coeff,
                               ext_rew_coeff=self.ext_coeff,
                               record_rollouts=self.use_recorder,
                               dynamics=dynamics, exp_name=self.exp_name, env_name=self.env_name,
                               video_log_freq=self.video_log_freq, model_save_freq=self.model_save_freq,
                               use_apples=self.use_apples, multi_envs=self.multi_envs, lstm=self.lstm, lstm1_size=self.lstm1_size, lstm2_size=self.lstm2_size)

        self.buf_advs = np.zeros((nenvs, self.rollout.nsteps), np.float32)
        self.buf_rets = np.zeros((nenvs, self.rollout.nsteps), np.float32)

        if self.normrew:
            self.rff = RewardForwardFilter(self.gamma)
            self.rff_rms = RunningMeanStd()

        self.step_count = 0
        self.t_last_update = time.time()
        self.t_start = time.time()

    def stop_interaction(self):
        for env in self.envs:
            env.close()

    def calculate_advantages(self, rews, use_news, gamma, lam):
        nsteps = self.rollout.nsteps
        lastgaelam = 0
        for t in range(nsteps - 1, -1, -1):  # nsteps-2 ... 0
            nextnew = self.rollout.buf_news[:, t + 1] if t + 1 < nsteps else self.rollout.buf_new_last
            if not use_news:
                nextnew = 0
            nextvals = self.rollout.buf_vpreds[:, t + 1] if t + 1 < nsteps else self.rollout.buf_vpred_last
            nextnotnew = 1 - nextnew
            delta = rews[:, t] + gamma * nextvals * nextnotnew - self.rollout.buf_vpreds[:, t]
            self.buf_advs[:, t] = lastgaelam = delta + gamma * lam * nextnotnew * lastgaelam
        self.buf_rets[:] = self.buf_advs + self.rollout.buf_vpreds

    def update(self):
        if self.normrew:
            rffs = np.array([self.rff.update(rew) for rew in self.rollout.buf_rews.T])
            rffs_mean, rffs_std, rffs_count = mpi_moments(rffs.ravel())
            self.rff_rms.update_from_moments(rffs_mean, rffs_std ** 2, rffs_count)
            rews = self.rollout.buf_rews / np.sqrt(self.rff_rms.var)
        else:
            rews = np.copy(self.rollout.buf_rews)
        self.calculate_advantages(rews=rews, use_news=self.use_news, gamma=self.gamma, lam=self.lam)

        info = dict(
            advmean=self.buf_advs.mean(),
            advstd=self.buf_advs.std(),
            retmean=self.buf_rets.mean(),
            retstd=self.buf_rets.std(),
            vpredmean=self.rollout.buf_vpreds.mean(),
            vpredstd=self.rollout.buf_vpreds.std(),
            ev=explained_variance(self.rollout.buf_vpreds.ravel(), self.buf_rets.ravel()),
            rew_mean=np.mean(self.rollout.buf_rews),
            recent_best_ext_ret=self.rollout.current_max
        )
        if self.rollout.best_ext_ret is not None:
            info['best_ext_ret'] = self.rollout.best_ext_ret

        # normalize advantages
        if self.normadv:
            m, s = get_mean_and_std(self.buf_advs)
            self.buf_advs = (self.buf_advs - m) / (s + 1e-7)
        envsperbatch = (self.nenvs * self.nsegs_per_env) // self.nminibatches
        envsperbatch = max(1, envsperbatch)
        envinds = np.arange(self.nenvs * self.nsegs_per_env)

        def mask(x):
            #print("x shape: {}".format(np.shape(x)))
            #pseudo_dones = self.rollout.buf_news
            #print("mask shape: {}".format(np.shape(pseudo_dones)))
            #done_mask = pseudo_dones == -1
            #no_grad_mask = done_mask.astype(int)
            #grad_mask = 1 - done_mask.astype(int)
            #no_grad_mask = tf.cast(no_grad_mask, x.dtype)
            #grad_mask = tf.cast(grad_mask, x.dtype)
            #no_grad_mask = 
            #result = tf.placeholder(x.dtype, shape=(sh[0], sh[1]) + sh[2:])
            #result = tf.stop_gradient(tf.multiply(tf.cast(no_grad_mask, x.dtype), x)) + tf.multiply(tf.cast(grad_mask, x.dtype), x)
            #print(tf.shape(result))
            #return result
            return x

        def resh(x):
            if self.nsegs_per_env == 1:
                return x
            sh = x.shape
            return x.reshape((sh[0] * self.nsegs_per_env, self.nsteps_per_seg) + sh[2:])

        print(self.rollout.buf_news)
        ph_buf = [
            (self.stochpol.ph_ac, mask(resh(self.rollout.buf_acs))),
            (self.ph_rews, mask(resh(self.rollout.buf_rews))),
            (self.ph_oldvpred, mask(resh(self.rollout.buf_vpreds))),
            (self.ph_oldnlp, mask(resh(self.rollout.buf_nlps))),
            (self.stochpol.ph_ob, mask(resh(self.rollout.buf_obs))),
            (self.ph_ret, mask(resh(self.buf_rets))),
            (self.ph_adv, mask(resh(self.buf_advs))),
        ]
        #print("Buff obs shape: {}".format(self.rollout.buf_obs.shape))
        #print("Buff rew shape: {}".format(self.rollout.buf_rews.shape))
        #print("Buff nlps shape: {}".format(self.rollout.buf_nlps.shape))
        #print("Buff vpreds shape: {}".format(self.rollout.buf_vpreds.shape))
        ph_buf.extend([
            (self.dynamics.last_ob,
             self.rollout.buf_obs_last.reshape([self.nenvs * self.nsegs_per_env, 1, *self.ob_space.shape]))
        ])
        mblossvals = []

        for _ in range(self.nepochs):
            np.random.shuffle(envinds)
            for start in range(0, self.nenvs * self.nsegs_per_env, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                fd = {ph: buf[mbenvinds] for (ph, buf) in ph_buf}
                fd.update({self.ph_lr: self.lr, self.ph_cliprange: self.cliprange})
                if self.lstm:
                    fd.update({
                        self.stochpol.c_in_1: self.rollout.train_lstm1_c[mbenvinds,:],
                        self.stochpol.h_in_1: self.rollout.train_lstm1_h[mbenvinds,:]
                    })
                if self.lstm and self.lstm2_size:
                    fd.update({
                        self.stochpol.c_in_2: self.rollout.train_lstm2_c[mbenvinds,:],
                        self.stochpol.h_in_2: self.rollout.train_lstm2_h[mbenvinds,:]
                    })

                mblossvals.append(getsess().run(self._losses + (self._train,), fd)[:-1])

        mblossvals = [mblossvals[0]]
        info.update(zip(['opt_' + ln for ln in self.loss_names], np.mean([mblossvals[0]], axis=0)))
        info["rank"] = MPI.COMM_WORLD.Get_rank()
        self.n_updates += 1
        info["n_updates"] = self.n_updates
        info.update({dn: (np.mean(dvs) if len(dvs) > 0 else 0) for (dn, dvs) in self.rollout.statlists.items()})
        info.update(self.rollout.stats)
        if "states_visited" in info:
            info.pop("states_visited")
        tnow = time.time()
        info["ups"] = 1. / (tnow - self.t_last_update)
        info["total_secs"] = tnow - self.t_start
        info['tps'] = MPI.COMM_WORLD.Get_size() * self.rollout.nsteps * self.nenvs / (tnow - self.t_last_update)
        self.t_last_update = tnow

        return info

    def step(self):
        #print("Collecting rollout")
        self.rollout.collect_rollout()
        #print("Performing update")
        update_info = self.update()
        return {'update': update_info}

    def get_var_values(self):
        return self.stochpol.get_var_values()

    def set_var_values(self, vv):
        self.stochpol.set_var_values(vv)

    def restore(self):
        # self.stochpol.vpred = tf.get_collection("vpred")[0]
        # self.stochpol.a_samp = tf.get_collection("a_samp")[0]
        # self.stochpol.entropy = tf.get_collection("entropy")[0]
        # self.stochpol.nlp_samp = tf.get_collection("nlp_samp")[0]
        # self.stochpol.ph_ob = tf.get_collection("ph_ob")[0]
        self.ph_adv = tf.get_collection("adv")[0]
        self.ph_ret = tf.get_collection("ret")[0]
        self.ph_rews = tf.get_collection("rews")[0]
        self.ph_oldnlp = tf.get_collection("oldnlp")[0]
        self.ph_oldvpred = tf.get_collection("oldvpred")[0]
        self.ph_lr = tf.get_collection("lr")[0]
        self.ph_cliprange = tf.get_collection("cliprange")[0]
        neglogpac = self.stochpol.pd.neglogp(self.stochpol.ph_ac)
        entropy = tf.get_collection("agent_entropy")[0]
        vpred = self.stochpol.vpred
        vf_loss = tf.get_collection("vf_loss")[0]
        ratio = tf.get_collection("ratio")[0]
        negadv = - self.ph_adv
        pg_losses1 = negadv * ratio
        pg_losses2 = tf.get_collection("pg_losses2")[0]
        pg_loss_surr = tf.get_collection("loss_surr")[0]
        pg_loss = tf.get_collection("pg_loss")[0]
        ent_loss = (- self.ent_coeff) * entropy
        approxkl = tf.get_collection("approxkl")[0]
        clipfrac = tf.get_collection("clipfrac")[0]

        self.total_loss = pg_loss + ent_loss + vf_loss
        self.to_report = {'tot': self.total_loss, 'pg': pg_loss, 'vf': vf_loss, 'ent': entropy,
                          'approxkl': approxkl, 'clipfrac': clipfrac}


class RewardForwardFilter(object):
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems
