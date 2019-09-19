import tensorflow as tf
import gym
import gym_deepmindlab
from wrappers import ProcessFrame84
from baselines.common.atari_wrappers import FrameStack
from baselines.common.distributions import make_pdtype
from utils import setup_tensorflow_session
from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim

class Evaluator(object):
    def __init__(self, env, num_episodes, exp_name):
        self.exp_name = exp_name
        self.env = env
        self.num_episodes = num_episodes

    def eval_model(self):
        saver = tf.train.Saver()
        saver.restore(sess, "/tmp/"+self.exp_name + ".ckpt")
        print("Model restored")
        for i in range(self.num_episodes):
            ob = self.env.reset()
            eprews = []
            for step in range(900):
                action, vpred, nlp = get_ac_value_nlp(ob)
                ob, rew, done, info = self.env.step(action)
                if rew is None:
                    eprews.append(0)
                else:
                    eprews.append(rew)
                if done:
                    print("Episode finished after {} timesteps".format(step))
                    print("Episode Reward is {}".format(sum(eprews)))
                    break
            print("Episode {} cumulative reward: {}".format(i, sum(eprews)))


    def get_ac_value_nlp(self, ob):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]




def start_eval(**args):
    print("Starting model evaluation")
    env = gym.make(args['env'])
    env = ProcessFrame84(env, crop=False)
    env = FrameStack(env, 4)
    num_episodes = args['num_episodes']
    exp_name = args['exp_name']
    evaluator = Evaluator(env, num_episodes, exp_name)
    evaluator.eval_model()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--env', type=str, default='DeepmindLabNavMazeStatic01-v0')
    parser.add_argument('--num_episodes', type=int, default=100)
    args = parser.parse_args()
    start_eval(**args.__dict__)


