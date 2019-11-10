import tensorflow as tf
import gym
import gym_deepmindlab
from wrappers import ProcessFrame84
from baselines.common.atari_wrappers import FrameStack
import numpy as np
import cv2
import os

def format_obs(obs_name, obs):
    nums = ",".join(map(str, obs))
    dict_format = "{" + nums + "}"
    final_str = "observation \"{}\" - {}\n".format(obs_name, dict_format)
    return final_str

def start_eval(**args):
    print("Starting model evaluation")
    env = gym.make(args['env'])
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    num_episodes = args['num_episodes']
    exp_name = args['exp_name']
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph("models/" + exp_name + ".ckpt" + ".meta")
        saver.restore(sess, "models/" + exp_name + ".ckpt")
        vpred = tf.get_collection("vpred")[0]
        a_samp = tf.get_collection("a_samp")[0]
        entropy = tf.get_collection("entropy")[0]
        nlp_samp = tf.get_collection("nlp_samp")[0]
        ph_ob = tf.get_collection("ph_ob")[0]
        print("Model restored")
        env_name = args['env']
        env_type = env_name.split('-')[0][-3:]
        if env_type == "New":
            rew_type = "sparse"
        elif env_type == "wno":
            rew_type = "no"
        else:
            rew_type = "dense"
        for i in range(num_episodes):
            trajectory_file = exp_name + "_eval_on_" + rew_type  + "_itr" + str(i) + "_trajectory.txt"
            if not os.path.exists("trajectories"):
                os.makedirs("trajectories")
            trajectory_path = os.path.join("trajectories", trajectory_file)
            ob = env.reset()
            ob = np.array(ob)
            ob = ob.reshape((1,84,84,4))
            eprews = []
            ep_images = []
            for step in range(200):
                if step == 0:
                    ep_images.append(env.unwrapped._last_observation)
                action, vp, nlp = sess.run([a_samp, vpred, nlp_samp],
                                           feed_dict={ph_ob: ob[:,None]})
                action = action[:,0]
                ob, rew, done, info = env.step(action[0])
                pos_trans, pos_rot, vel_trans, vel_rot = env.unwrapped.get_pos_and_vel()
                ob = np.array(ob).reshape((1,84,84,4))
                ep_images.append(env.unwrapped._last_observation)
                with open(trajectory_path, 'a') as f:
                    f.write(format_obs("DEBUG.POS.TRANS", pos_trans))
                    f.write(format_obs("DEBUG.POS.ROT", pos_rot))
                    f.write(format_obs("VEL.TRANS", vel_trans))
                    f.write(format_obs("VEL.ROT", vel_rot))
                if rew is None:
                    eprews.append(0)
                elif rew == 10:
                    print("Reached goal on step {}".format(step))
                    eprews.append(rew)
                    break
                else:
                    eprews.append(rew)
            for j in range(len(ep_images)):
                image_file = os.path.join("images", exp_name + "_eval_on_" + rew_type + "_itr" + str(i) + "_{}".format(j) + ".png")
                cv2.imwrite(image_file, ep_images[j])
            print("Total reward is {}".format(sum(eprews)))




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='DeepmindLabNavMazeStatic01-v0')
    parser.add_argument('--num_episodes', type=int, default=1)
    args = parser.parse_args()
    start_eval(**args.__dict__)


