import tensorflow.compat.v1 as tf
from baselines.common.distributions import make_pdtype
import os

from utils import getsess, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim

tf.disable_v2_behavior()
class CnnPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, scope="policy"):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            x = flatten_two_dims(self.ph_ob)
            self.flat_features = self.get_features(x, reuse=tf.AUTO_REUSE)
            self.features = unflatten_first_dim(self.flat_features, sh)

            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                x = fc(self.flat_features, units=hidsize, activation=activ)
                x = fc(x, units=hidsize, activation=activ)
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)
                vpred = fc(x, name='value_function_output', units=1, activation=None)
            pdparam = unflatten_first_dim(pdparam, sh)
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
            self.a_samp = pd.sample()
            self.entropy = pd.entropy()
            self.nlp_samp = pd.neglogp(self.a_samp)
            tf.add_to_collection('ph_ac', self.ph_ac)
            tf.add_to_collection('vpred', self.vpred)
            tf.add_to_collection('pdparam', pdparam)
            tf.add_to_collection('a_samp', self.a_samp)
            tf.add_to_collection('entropy', self.entropy)
            tf.add_to_collection('nlp_samp', self.nlp_samp)
            tf.add_to_collection('ph_ob', self.ph_ob)

    def get_features(self, x, reuse):
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        return x

    def get_ac_value_nlp(self, ob):
        a, vpred, nlp = \
            getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ob[:, None]})
        return a[:, 0], vpred[:, 0], nlp[:, 0]

    def get_ac_value_nlp_eval(self, ob):
        a, vpred, nlp = getsess().run([self.a_samp, self.vpred, self.nlp_samp],
                          feed_dict={self.ph_ob: ((ob,),)})
        return a[:,0], vpred[:,0], nlp[:,0]

    def save_model(self, model_name, ep_num):
        self.saver = tf.train.Saver()
        if not os.path.exists("models"):
            os.makedirs("models")
        if ep_num:
            path = "models/"+model_name+ "_ep{}".format(ep_num) + ".ckpt"
        else:
            path = "models/"+model_name+ "_{}".format("final") + ".ckpt"
        self.saver.save(getsess(), path)
        print("Model saved to path",path)

    def restore_model(self, model_name):
        saver = tf.train.import_meta_graph("models/" + model_name + ".ckpt" + ".meta")
        saver.restore(getsess(), "models/" + model_name + ".ckpt")
        self.vpred = tf.get_collection("vpred")[0]
        self.a_samp = tf.get_collection("a_samp")[0]
        self.entropy = tf.get_collection("entropy")[0]
        self.nlp_samp = tf.get_collection("nlp_samp")[0]
        self.ph_ob = tf.get_collection("ph_ob")[0]

    def restore(self):
        self.vpred = tf.get_collection("vpred")[0]
        self.a_samp = tf.get_collection("a_samp")[0]
        self.entropy = tf.get_collection("entropy")[0]
        self.nlp_samp = tf.get_collection("nlp_samp")[0]
        self.ph_ob = tf.get_collection("ph_ob")[0]
        self.ph_ac = tf.get_collection("ph_ac")[0]
        pdparam = tf.get_collection("pdparam")[0]
        self.pd = pd = self.ac_pdtype.pdfromflat(pdparam)
        
