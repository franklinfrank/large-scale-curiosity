import tensorflow.compat.v1 as tf
import gym
import gym_deepmindlab
from wrappers import ProcessFrame84
from baselines.common.atari_wrappers import FrameStack
import numpy as np
import cv2
import os
import sys
from baselines.common import set_global_seeds
from gym.utils.seeding import hash_seed

tf.disable_v2_behavior()
def format_obs(obs_name, obs):
    nums = ",".join(map(str, obs))
    dict_format = "{" + nums + "}"
    final_str = "observation \"{}\" - {}\n".format(obs_name, dict_format)
    return final_str

def start_eval(**args):
        print("Starting model evaluation")
        env = gym.make(args['env'])
        #env = ProcessFrame84(env, crop=False)
        #env = FrameStack(env, 4)
        num_episodes = args['num_episodes']
        exp_name = args['exp_name']
        save_name = args['save_name']
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        num_steps = args['num_steps']
        aux_input = args['aux_input']
        #process_seed = args["seed"] + 1000 * MPI.COMM_WORLD.Get_rank()
        #process_seed = hash_seed(process_seed, max_bytes=4)
        #set_global_seeds(process_seed)
        #setup_mpi_gpus()

        with tf.Session(config=config) as sess:
                saver = tf.train.import_meta_graph("models/" + exp_name + ".ckpt" + ".meta")
                saver.restore(sess, "models/" + exp_name + ".ckpt")
                vpred = tf.get_collection("vpred")[0]
                a_samp = tf.get_collection("a_samp")[0]
                print(a_samp)
                entropy = tf.get_collection("entropy")[0]
                nlp_samp = tf.get_collection("nlp_samp")[0]
                ph_ob = tf.get_collection("ph_ob")[0]
                print(ph_ob)
                if args['lstm']:
                    print(tf.get_collection("c_out_1"))
                    c_out_1 = tf.get_collection("c_out_1")[0]
                    print(c_out_1)
                    h_out_1 = tf.get_collection("h_out_1")[0]
                    c_in_1 = tf.get_collection('c_in_1')[0]
                    print(c_in_1)
                    h_in_1 = tf.get_collection('h_in_1')[0]
                    if args['lstm2_size']:
                        c_out_2 = tf.get_collection('c_out_2')[0]
                        print(c_out_2)
                        h_out_2 = tf.get_collection('h_out_2')[0]
                        print(h_out_2)
                        c_in_2 = tf.get_collection('c_in_2')[0]
                        h_in_2 = tf.get_collection('h_in_2')[0]
                if aux_input:
                    ph_vel = tf.get_collection('ph_vel')[0]
                    ph_prev_ac = tf.get_collection('ph_prev_ac')[0]
                    ph_prev_rew = tf.get_collection('ph_prev_rew')[0]
                print("Model restored")
                env_name = args['env']
                env_type = env_name.split('-')[0][-3:]
                if env_type == "New":
                        rew_type = "sparse"
                elif env_type == "wno":
                        rew_type = "no"
                else:
                        rew_type = "dense"
                rew_type = env_name
                success_count = 0
                total_rew = 0
                for i in range(num_episodes):
                        trajectory_file = save_name + "_eval_on_" + rew_type  + "_itr" + str(i) + "_trajectory.txt"
                        if not os.path.exists("trajectories"):
                                os.makedirs("trajectories")
                        trajectory_path = os.path.join("trajectories", trajectory_file)
                        ob = env.reset()
                        ob = np.array(ob)
                        #ob = ob.reshape((1,84,84,4))
                        ob = ob.reshape((1,84,84,3))
                        eprews = []
                        ep_images = []
                        if args['lstm']:
                            lstm1_c = np.zeros((1, args['lstm1_size']))
                            lstm1_h = np.zeros((1, args['lstm1_size']))
                            if args['lstm2_size']:
                                lstm2_c = np.zeros((1, args['lstm2_size']))
                                lstm2_h = np.zeros((1, args['lstm2_size']))
                        success = 0
                        for step in range(num_steps):
                                #print(step)
                                if step == 0:
                                        ep_images.append(env.unwrapped._last_observation)
                                        if aux_input:
                                            prev_ac = np.zeros((1,1))
                                            vels = np.zeros((1,1,6))
                                            prev_rew = np.zeros((1,1))
                                if args['lstm']:
                                    feed_dict = {ph_ob: ob[:, None], c_in_1: lstm1_c, h_in_1: lstm1_h}
                                    if args['lstm2_size'] > 0:
                                        feed_dict.update({c_in_2: lstm2_c, h_in_2: lstm2_h})
                                        if aux_input:
                                            feed_dict.update({ph_prev_rew:prev_rew, ph_vel: vels, ph_prev_ac: prev_ac}) 
                                        action, vp, nlp, lstm1_c, lstm1_h, lstm2_c, lstm2_h = sess.run([a_samp, vpred, nlp_samp, c_out_1, h_out_1, \
                                            c_out_2, h_out_2], feed_dict=feed_dict)
                                        
                                    else:
                                        action, vp, nlp, lstm1_c, lstm1_h = \
                                            sess.run([a_samp, vpred, nlp_samp, c_out_1, h_out_1], feed_dict=feed_dict)

                                else:
                                    action, vp, nlp = sess.run([a_samp, vpred, nlp_samp], 
                                                feed_dict={ph_ob: ob[:,None]})
                                action = action[:,0]
                                #print(action)
                                ob, rew, done, info = env.step(action[0])
                                #print(rew)
                                #print(ob)
                                #print(info)
                                if not done:
                                    pos_trans, pos_rot, vel_trans, vel_rot = env.unwrapped.get_pos_and_vel()
                                if aux_input:
                                    #info_vel_trans = info['vel_trans']
                                    #info_vel_rot = info['vel_rot']
                                    #info_vels = [[np.append(info_vel_trans, info_vel_rot)]]
                                    #print("Info vels: {}".format(info_vels))
                                    vels = [[np.append(vel_trans, vel_rot)]]
                                    #print("Func vels: {}".format(vels))
                                    prev_rew = [[rew]]
                                    prev_ac = [action]
                                ob = np.array(ob).reshape((1,84,84,3))
                                #ob = np.array(ob).reshape((1,84,84,4))
                                ep_images.append(env.unwrapped._last_observation)
                                with open(trajectory_path, 'a') as f:
                                        f.write(format_obs("DEBUG.POS.TRANS", pos_trans))
                                        f.write(format_obs("DEBUG.POS.ROT", pos_rot))
                                        f.write(format_obs("VEL.TRANS", vel_trans))
                                        f.write(format_obs("VEL.ROT", vel_rot))
                                if rew is None:
                                        eprews.append(0)
                                elif rew >= 10:
                                        print("Reached goal on step {}".format(step))
                                        eprews.append(rew)
                                        success = 1
                                        break
                                        #ob = env.reset()
                                        #ob = np.array(ob)
                                        #ob = ob.reshape((1,84,84,4))
                                        #ob = ob.reshape((1,84,84,3))
                                else:
                                        eprews.append(rew)
                        for j in range(len(ep_images)):
                                image_file = os.path.join("images", save_name + "_eval_on_" + rew_type + "_itr" + str(i) + "_{}".format(j) + ".png")
                                cv2.imwrite(image_file, ep_images[j])                           
                        print("Total reward is {}".format(sum(eprews)))
                        success_count += success
                        total_rew += sum(eprews)
                print("Success rate: {}".format(success_count * 1.0 / num_episodes))
                print("Avg reward: {}".format(total_rew * 1.0 / num_episodes))
                print("Eval name: {}".format(save_name + "_eval_on_" + rew_type))
                with open("results/" + save_name + "_eval_on_" + rew_type + "_eval_results.txt", "w") as f:
                    f.write("Number of episodes: {}\n".format(num_episodes))
                    f.write("Success rate: {}\n".format(success_count * 1.0 / num_episodes))
                    f.write("Avg reward: {}\n".format(total_rew * 1.0 /  num_episodes))


if __name__ == "__main__":
        import argparse
        import os 
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--exp_name', type=str, default='')
        parser.add_argument('--save_name', type=str,default='')
        parser.add_argument('--seed', type=int, default=0)
        parser.add_argument('--env', type=str, default='DeepmindLabNavMazeStatic01-v0')
        parser.add_argument('--num_episodes', type=int, default=1)
        parser.add_argument('--num_steps', type=int, default=200)
        parser.add_argument('--lstm', type=int, default=0)
        parser.add_argument('--lstm1_size', type=int, default=512)
        parser.add_argument('--lstm2_size', type=int, default=0)
        parser.add_argument('--aux_input', type=int, default=0)
        args = parser.parse_args()
        start_eval(**args.__dict__)
