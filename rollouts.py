from collections import deque, defaultdict

import numpy as np
from mpi4py import MPI

from recorder import Recorder
import os
import cv2
from evaluator import Evaluator


class Rollout(object):
    def __init__(self, ob_space, ac_space, nenvs, nsteps_per_seg, nsegs_per_env, nlumps, envs, policy, int_rew_coeff, ext_rew_coeff, \
                   record_rollouts, dynamics, exp_name, env_name, video_log_freq, model_save_freq, use_apples, multi_envs=None, lstm=False, lstm1_size=512, lstm2_size=0, depth_pred=0, early_stop=0):
        self.nenvs = nenvs
        self.nsteps_per_seg = nsteps_per_seg
        self.nsegs_per_env = nsegs_per_env
        self.nsteps = self.nsteps_per_seg * self.nsegs_per_env
        self.ob_space = ob_space
        self.ac_space = ac_space
        self.nlumps = nlumps
        self.lump_stride = nenvs // self.nlumps
        self.envs = envs
        self.policy = policy
        self.dynamics = dynamics
        self.exp_name = exp_name
        self.env_name = env_name
        self.model_save_freq = model_save_freq
        self.video_log_freq = video_log_freq
        self.model_save_freq = model_save_freq
        self.lstm = lstm
        self.lstm1_size = lstm1_size
        self.lstm2_size = lstm2_size
        self.depth_pred = depth_pred
#        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff * np.clip(ext_rew, -1., 1.) + int_rew_coeff * int_rew
        self.reward_fun = lambda ext_rew, int_rew: ext_rew_coeff*ext_rew + int_rew_coeff*int_rew
        self.evaluator = Evaluator(env_name, 1, exp_name, policy)
        self.buf_vpreds = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_nlps = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_vels = np.empty((nenvs, self.nsteps, 6), np.float32)
        self.buf_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_prev_ext_rews = np.empty((nenvs, self.nsteps), np.float32)
        self.buf_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.buf_prev_acs = np.empty((nenvs, self.nsteps, *self.ac_space.shape), self.ac_space.dtype)
        self.prev_ac_ph = np.zeros((nenvs, *self.ac_space.shape), self.ac_space.dtype)
        #if not self.depth_pred:
        self.buf_obs = np.empty((nenvs, self.nsteps, *self.ob_space.shape), self.ob_space.dtype)
        self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, *self.ob_space.shape), np.float32)
        #else:
        #self.buf_obs = np.empty((nenvs, self.nsteps, 84, 84, 3), self.ob_space.dtype)
        #self.buf_obs_last = np.empty((nenvs, self.nsegs_per_env, 84, 84, 3), np.float32)
        #self.num_actions = self.ac_space.shape[0]
        self.early_stop= early_stop
        if self.depth_pred:
            self.buf_depths = np.empty((nenvs, self.nsteps, 64))
        if early_stop:
            self.early_stops = np.zeros(nenvs)
            self.grad_mask = np.empty((nenvs, self.nsteps))
        self.buf_news = np.zeros((nenvs, self.nsteps), np.float32)
        self.buf_new_last = self.buf_news[:, 0, ...].copy()
        self.buf_vpred_last = self.buf_vpreds[:, 0, ...].copy()
        self.depth_pred = depth_pred
        self.env_results = [None] * self.nlumps
        # self.prev_feat = [None for _ in range(self.nlumps)]
        # self.prev_acs = [None for _ in range(self.nlumps)]
        self.int_rew = np.zeros((nenvs,), np.float32)

        self.recorder = Recorder(nenvs=self.nenvs, nlumps=self.nlumps) if record_rollouts else None
        self.statlists = defaultdict(lambda: deque([], maxlen=100))
        self.stats = defaultdict(float)
        self.best_ext_ret = None
        self.all_visited_rooms = []
        self.all_scores = []
        self.multi_envs = multi_envs

        self.step_count = 0
        self.use_apples = use_apples

    def collect_rollout(self):
        self.ep_infos_new = []
        #for i in range(self.nlumps):
            #self.env_reset(i)
        for t in range(self.nsteps):
            self.rollout_step()
        self.calculate_reward()
        self.update_info()

    def calculate_reward(self):
        int_rew = self.dynamics.calculate_loss(ob=self.buf_obs,
                                               last_ob=self.buf_obs_last,
                                               acs=self.buf_acs)
        self.buf_rews[:] = self.reward_fun(int_rew=int_rew, ext_rew=self.buf_ext_rews)

    def depth_process(self, rgbds):
        rgb_list = []
        d_list = []
        for rgbd in rgbds:
            rgb = rgbd[:,:,0:3]
            d = rgbd[:,:,3]
            d = d[16:-16,:]
            d = d[:,2:-2]
            d = d[::13,::5]
            d = d.flatten()
            d = np.power(d/255.0, 10)
            d = np.digitize(d,[0,0.05,0.175,0.3,0.425,0.55,0.675,0.8,1.01])
            d -= 1
            rgb_list.append(rgb)
            d_list.append(d)
        return np.array(rgb_list), np.array(d_list)

    def rollout_step(self):
        t = self.step_count % self.nsteps
        s = t % self.nsteps_per_seg
        ep_num = self.step_count // self.nsteps_per_seg
        for l in range(self.nlumps):
            obs, prevrews, news, infos = self.env_get(l)
            if self.depth_pred:
                obs, depths = self.depth_process(obs)
                #print(np.shape(obs))
                #print(np.shape(depths))
                
            if self.lstm:
                for idx in range(len(news)):
                    if news[idx]:
                        self.policy.lstm1_c[idx] = np.zeros(self.lstm1_size)
                        self.policy.lstm1_h[idx] = np.zeros(self.lstm1_size)
                        if self.lstm2_size:
                            self.policy.lstm2_c[idx] = np.zeros(self.lstm2_size)
                            self.policy.lstm2_h[idx] = np.zeros(self.lstm2_size)
            if s == 0 and self.lstm:
                self.train_lstm1_c = self.policy.lstm1_c 
                self.train_lstm1_h = self.policy.lstm1_h
                if self.lstm2_size:
                    self.train_lstm2_c = self.policy.lstm2_c 
                    self.train_lstm2_h = self.policy.lstm2_h


            if l == 0 and self.video_log_freq > 0 and ep_num % self.video_log_freq == 0:
                zero_env_obs = self.envs[0].get_latest_ob()
                dirname = os.path.abspath(os.path.dirname(__file__))
                image_folder = 'images'
                image_file = os.path.join(dirname, image_folder + "/" + self.exp_name +"_train_ep" + str(ep_num)+"_{}".format(s) + ".png")
                cv2.imwrite(image_file, zero_env_obs)
                
            # if t > 0:
            #     prev_feat = self.prev_feat[l]
            #     prev_acs = self.prev_acs[l]
            if prevrews is not None:
                if self.use_apples:
                    prevrews = [x if x is not None else 0 for x in prevrews]
                else:
                    prevrews = [x if x is not None and x >= 10 else 0 for x in prevrews]
            else:
                prevrews = np.zeros(self.nenvs)
            #print(prevrews)
            vels = []
            for info in infos:
                #print("info: {}".format(info))
                epinfo = info.get('episode', {})
                mzepinfo = info.get('mz_episode', {})
                retroepinfo = info.get('retro_episode', {})
                epinfo.update(mzepinfo)
                epinfo.update(retroepinfo)
                if 'vel_trans' in info:
                    step_vel = np.concatenate((info['vel_trans'], info['vel_rot']), axis=None)
                    vels.append(step_vel)
                else:
                    vels.append(np.zeros(6))
                
                if epinfo:
                    #print("epinfo: {}".format(epinfo))
                    if "n_states_visited" in info:
                        epinfo["n_states_visited"] = info["n_states_visited"]
                        epinfo["states_visited"] = info["states_visited"]
                    if "multi_envs" in info:
                        for env in self.multi_envs:
                            rew_key = env + "_reward"
                            #cov_key = env + "_coverage"
                            if rew_key in info:
                                epinfo[rew_key] = info[rew_key]
                                #epinfo[cov_key] = info[cov_key]
                    if 'found' in info:
                        epinfo['found'] = info['found']
                        #print("updated found")
                    if 'visit_count' in info:
                        epinfo['visit_count'] = info['visit_count']
                
                    self.ep_infos_new.append((self.step_count, epinfo))
            #print(vels)
            if len(vels) == 0:
                vels = np.zeros((self.lump_stride, 6), np.float32)
            sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
            if t > 0:
                prev_acs = self.buf_acs[sli, t-1]
            else:
                prev_acs = self.prev_ac_ph
            for i in range(len(news)):
                if news[i]:
                    prev_acs[i] = 0

            if self.depth_pred:
                acs, vpreds, nlps = self.policy.get_ac_value_nlp_extra_input(obs, vels, prev_acs, prevrews)
            else:
                acs, vpreds, nlps = self.policy.get_ac_value_nlp(obs)
            self.env_step(l, acs)

            # self.prev_feat[l] = dyn_feat
            # self.prev_acs[l] = acs
            if self.early_stop:
                for i in range(len(prevrews)):
                    rew = prevrews[i]
                    if rew >= 10:
                        self.early_stops[i] = 1
                    if news[i]:
                        self.early_stops[i] = 0
            if self.early_stop:
                self.grad_mask[sli, t] = self.early_stops
            self.buf_obs[sli, t] = obs
            self.buf_news[sli, t] = news
            self.buf_vpreds[sli, t] = vpreds
            self.buf_nlps[sli, t] = nlps
            self.buf_acs[sli, t] = acs
            self.buf_prev_acs[sli, t] = prev_acs
            self.buf_vels[sli, t] = vels
            self.buf_prev_ext_rews[sli, t] = prevrews
            if self.depth_pred:
                self.buf_depths[sli, t] = depths
            if t > 0:
                self.buf_ext_rews[sli, t - 1] = prevrews
            if s == self.nsteps_per_seg - 1:
                self.prev_ac_ph[sli] = acs
            # if t > 0:
            #     dyn_logp = self.policy.call_reward(prev_feat, pol_feat, prev_acs)
            #
            #     int_rew = dyn_logp.reshape(-1, )
            #
            #     self.int_rew[sli] = int_rew
            #     self.buf_rews[sli, t - 1] = self.reward_fun(ext_rew=prevrews, int_rew=int_rew)
            if self.recorder is not None:
                self.recorder.record(timestep=self.step_count, lump=l, acs=acs, infos=infos, int_rew=self.int_rew[sli],
                                     ext_rew=prevrews, news=news)
        self.step_count += 1
        if s == self.nsteps_per_seg - 1:
            for l in range(self.nlumps):
                sli = slice(l * self.lump_stride, (l + 1) * self.lump_stride)
                nextobs, ext_rews, nextnews, newinfos = self.env_get(l)
                if self.depth_pred:
                    nextobs, nextdepths = self.depth_process(nextobs)
                if ext_rews is not None:
                    ext_rews = [x if x is not None else 0 for x in ext_rews]
                self.buf_obs_last[sli, t // self.nsteps_per_seg] = nextobs
                if t == self.nsteps - 1:
                    self.buf_new_last[sli] = nextnews
                    self.buf_ext_rews[sli, t] = ext_rews
                    newvels = []
                    for vel_info in newinfos:
                        if 'vel_trans' in vel_info:
                            newvels.append(np.concatenate([vel_info['vel_trans'], vel_info['vel_rot']], axis=0))
                        else:
                            newvels.append(np.zeros(6)) 
                    #if len(newvels) == 0:
                        #newvels = np.zeros((self.lump_stride, 6), np.float32) 
                    if self.depth_pred: 
                        _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp_extra_input(nextobs, newvels, acs, ext_rews)
                    else:
                        _, self.buf_vpred_last[sli], _ = self.policy.get_ac_value_nlp(nextobs)
                    # dyn_logp = self.policy.call_reward(self.prev_feat[l], last_pol_feat, prev_acs)
                    # dyn_logp = dyn_logp.reshape(-1, )
                    # int_rew = dyn_logp
                    #
                    # self.int_rew[sli] = int_rew
                    # self.buf_rews[sli, t] = self.reward_fun(ext_rew=ext_rews, int_rew=int_rew)
            if self.video_log_freq > 0 and ep_num % self.video_log_freq == 0:
                self.evaluator.eval_model(ep_num)
            if self.model_save_freq > 0 and ep_num % self.model_save_freq == 0:
                self.policy.save_model(self.exp_name, ep_num)
            print("Update {}".format(ep_num))

    def update_info(self):
        all_ep_infos = MPI.COMM_WORLD.allgather(self.ep_infos_new)
        all_ep_infos = sorted(sum(all_ep_infos, []), key=lambda x: x[0])
        if all_ep_infos:
            print(all_ep_infos[0][1])
            all_ep_infos = [i_[1] for i_ in all_ep_infos]  # remove the step_count
            keys_ = all_ep_infos[0].keys()
            all_ep_infos = {k: [i[k] for i in all_ep_infos] for k in keys_}

            self.statlists['eprew'].extend(all_ep_infos['r'])
            self.stats['eprew_recent'] = np.mean(all_ep_infos['r'])
            self.statlists['eplen'].extend(all_ep_infos['l'])
            self.stats['epcount'] += len(all_ep_infos['l'])
            self.stats['tcount'] += sum(all_ep_infos['l'])
            if 'visited_rooms' in keys_:
                # Montezuma specific logging.
                self.stats['visited_rooms'] = sorted(list(set.union(*all_ep_infos['visited_rooms'])))
                self.stats['pos_count'] = np.mean(all_ep_infos['pos_count'])
                self.all_visited_rooms.extend(self.stats['visited_rooms'])
                self.all_scores.extend(all_ep_infos["r"])
                self.all_scores = sorted(list(set(self.all_scores)))
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited rooms")
                    print(self.all_visited_rooms)
                    print("All scores")
                    print(self.all_scores)
            if 'levels' in keys_:
                # Retro logging
                temp = sorted(list(set.union(*all_ep_infos['levels'])))
                self.all_visited_rooms.extend(temp)
                self.all_visited_rooms = sorted(list(set(self.all_visited_rooms)))
                if MPI.COMM_WORLD.Get_rank() == 0:
                    print("All visited levels")
                    print(self.all_visited_rooms)
            if 'found' in keys_:
                self.stats['found_frac'] = np.mean(all_ep_infos['found'])
                print("Found frac: {}".format(self.stats['found_frac']))
            if 'visit_count' in keys_:
                self.stats['visit_count'] = np.mean(all_ep_infos['visit_count'])
                print("Visit count: {}".format(self.stats['visit_count']))
            #if self.multi_envs:
                #for env in self.multi_envs:
                    #rew_key = env + "_reward"
                    #cov_key = env + "_coverage"
                    #env_rews = [i[rew_key] for i in all_ep_infos if rew_key in i]
                    #env_covs = [i[cov_key] for i in all_ep_infos if cov_key in i]
                    #self.statlists['eprew_' + env].extend(env_rews)
                    #self.stats['eprew_recent_' + env] = np.mean(env_rews)
                    #self.stats['coverage_' + env] = np.mean(env_covs)
            current_max = np.max(all_ep_infos['r'])
        else:
            current_max = None
        self.ep_infos_new = []

        if current_max is not None:
            if (self.best_ext_ret is None) or (current_max > self.best_ext_ret):
                self.best_ext_ret = current_max
        self.current_max = current_max

    def env_step(self, l, acs):
        self.envs[l].step_async(acs)
        self.env_results[l] = None

    def env_reset(self, l):
        ob = self.envs[l].reset()
        out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        return out

    def env_get(self, l):
        if self.step_count == 0:
            ob = self.envs[l].reset()
            out = self.env_results[l] = (ob, None, np.ones(self.lump_stride, bool), {})
        else:
            if self.env_results[l] is None:
                out = self.env_results[l] = self.envs[l].step_wait()
            else:
                out = self.env_results[l]
        return out
        
