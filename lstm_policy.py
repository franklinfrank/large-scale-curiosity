import tensorflow as tf
from baselines.common.distributions import make_pdtype
import numpy as np
import os 

from utils import getsess, lstm, small_convnet, activ, fc, flatten_two_dims, unflatten_first_dim

class LSTMPolicy(object):
    def __init__(self, ob_space, ac_space, hidsize, batchsize,
                 ob_mean, ob_std, feat_dim, layernormalize, nl, lstm1_size, lstm2_size, scope="policy", depth_pred=0, aux_input=0):
        if layernormalize:
            print("Warning: policy is operating on top of layer-normed features. It might slow down the training.")
        self.layernormalize = layernormalize
        self.nl = nl
        self.ob_mean = ob_mean
        self.ob_std = ob_std
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            self.lstm1_size = lstm1_size
            self.lstm2_size = lstm2_size
            self.depth_pred = depth_pred
            self.aux_input = aux_input
            self.ob_space = ob_space
            self.ac_space = ac_space
            self.ac_pdtype = make_pdtype(ac_space)
            #print(ac_space.shape)
            self.num_actions = ac_space.n
            #if self.depth_pred:
             #   self.ph_ob = tf.placeholder(dtype=tf.int32, shape=(None, None, 84, 84, 3), name='ob')
            self.ph_ob = tf.placeholder(dtype=tf.int32,
                                        shape=(None, None) + ob_space.shape, name='ob')
            self.ph_ac = self.ac_pdtype.sample_placeholder([None, None], name='ac')
            if self.depth_pred:
                self.ph_depths = tf.placeholder(dtype=tf.int32, shape=(None, None, 64))
            if self.aux_input:
                self.ph_vel = tf.placeholder(dtype=tf.float32, shape=(None, None, 6), name='prev_vel')
                self.ph_prev_ac = tf.placeholder(dtype=tf.int32, shape=(None, None), name='prev_ac')
                self.ph_prev_rew = tf.placeholder(dtype=tf.float32, shape=(None, None), name='prev_rew')

            self.pd = self.vpred = None
            self.hidsize = hidsize
            self.feat_dim = feat_dim
            self.scope = scope
            pdparamsize = self.ac_pdtype.param_shape()[0]

            sh = tf.shape(self.ph_ob)
            if self.depth_pred:
                depth_sh = tf.shape(self.ph_depths)
            self.lstm_features = self.get_lstm_features(self.ph_ob, reuse=tf.AUTO_REUSE)
            print("Input shape into LSTM layer: {}".format(self.lstm_features.get_shape()))
            self.lstm1_c = np.zeros((batchsize, self.lstm1_size))
            self.lstm1_h = np.zeros((batchsize, self.lstm1_size))
            if self.lstm2_size > 0:
                self.lstm2_c = np.zeros((batchsize, self.lstm2_size))
                self.lstm2_h = np.zeros((batchsize, self.lstm2_size))
            self.lstm1_c_eval = np.zeros((1, self.lstm1_size))
            self.lstm1_h_eval = np.zeros((1, self.lstm1_size))
            if self.lstm2_size> 0:
                self.lstm2_c_eval = np.zeros((1, self.lstm2_size))
                self.lstm2_h_eval = np.zeros((1, self.lstm2_size))
            with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
                self.c_in_1 = tf.placeholder(tf.float32, name='c_1', shape=[None, self.lstm1_size])
                self.h_in_1 = tf.placeholder(tf.float32, name='h_1', shape=[None, self.lstm1_size])
                if self.lstm2_size:
                    self.c_in_2 = tf.placeholder(tf.float32, name='c_2', shape=[None, self.lstm2_size])
                    self.h_in_2 = tf.placeholder(tf.float32, name='h_2', shape=[None, self.lstm2_size])
                init_1 = tf.contrib.rnn.LSTMStateTuple(self.c_in_1, self.h_in_1)
                if self.lstm2_size:
                    init_2 = tf.contrib.rnn.LSTMStateTuple(self.c_in_2, self.h_in_2)
                if self.aux_input:
                    prev_rews = tf.expand_dims(self.ph_prev_rew, -1)
                    x = tf.concat([self.lstm_features, prev_rews], -1)
                else:
                    x = self.lstm_features
                x, self.c_out_1, self.h_out_1 = lstm(self.lstm1_size)(x, initial_state=init_1)
                if self.lstm2_size:
                    if self.aux_input:
                        prev_acs = tf.one_hot(self.ph_prev_ac, depth=self.num_actions)
                        x = tf.concat([x, tf.cast(prev_acs, tf.float32)], -1)
                        x = tf.concat([x, self.ph_vel], -1)

                    x, self.c_out_2, self.h_out_2  = lstm(self.lstm2_size)(x, initial_state=init_2)
                #x = lstm(256)(x)
                #x = lstm(hidsize)(self.lstm_features)
                print("Lstm output shape: {}".format(x.get_shape()))
                x = flatten_two_dims(x)
                pdparam = fc(x, name='pd', units=pdparamsize, activation=None)
                print("Pdparam shape: {}".format(pdparam.get_shape()))
                vpred = fc(x, name='value_function_output', units=1, activation=None)
                print("Vpred shape: {}".format(vpred.get_shape()))
                pdparam = unflatten_first_dim(pdparam, sh)
                # Depth prediction
                if self.depth_pred:
                    dpred = fc(x, name='depth_pred', units=128, activation=None)
                    dpred = [fc(dpred, name="depth_pred_pixel_{}".format(i), units=8, activation=None) for i in range(64)]
            if self.depth_pred:
                true_ds = flatten_two_dims(self.ph_depths)
                #d2 = tf.reshape(dpred, [-1, 64, 8])
                d2 = tf.transpose(dpred, [1,0,2])
                self.depth_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=d2, labels=true_ds)
                self.depth_loss = tf.reduce_mean(self.depth_loss, -1)
                self.depth_loss = tf.reduce_mean(self.depth_loss, -1)
            
            self.vpred = unflatten_first_dim(vpred, sh)[:, :, 0]
            #self.vpred = vpred
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
            tf.add_to_collection('c_out_1', self.c_out_1)
            tf.add_to_collection('h_out_1', self.h_out_1)
            tf.add_to_collection('c_in_1', self.c_in_1)
            tf.add_to_collection('h_in_1', self.h_in_1)
            if self.aux_input:
                tf.add_to_collection('ph_vel', self.ph_vel)
                tf.add_to_collection('ph_prev_ac', self.ph_prev_ac)
                tf.add_to_collection('ph_prev_rew', self.ph_prev_rew)
            if self.depth_pred:
                tf.add_to_collection('depth_loss_policy', self.depth_loss)
                tf.add_to_collection('ph_depths', self.ph_depths)
            if self.lstm2_size > 0:
                tf.add_to_collection('c_out_2', self.c_out_2)
                tf.add_to_collection('h_out_2', self.h_out_2)
                tf.add_to_collection('c_in_2', self.c_in_2)
                tf.add_to_collection('h_in_2', self.h_in_2)


    # ob has shape (batch_size, steps, 84, 84, 4)
    def get_lstm_features(self, x, reuse):
        # output_features = []
        # timesteps = tf.shape(ob)[1]
        # for i in range(timesteps):
        #     batch_size = tf.shape(ob)[0]
        #     ob_slice = tf.slice(ob, [0, i, 0, 0, 0], [-1, 1, -1, -1, -1])
        #     x = tf.reshape(ob_slice, [batch_size] + list(self.ob_space.shape))
        #     with tf.variable_scope(self.scope + "_features", reuse=reuse):
        #         x = (tf.to_float(x) - self.ob_mean) / self.ob_std
        #         x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)
        #     output_features.append(x)
        # return tf.stack(output_features, axis=1)
        x_has_timesteps = (x.get_shape().ndims == 5)
        if x_has_timesteps:
            sh = tf.shape(x)
            x = flatten_two_dims(x)

        with tf.variable_scope(self.scope + "_features", reuse=reuse):
            x = (tf.to_float(x) - self.ob_mean) / self.ob_std
            x = small_convnet(x, nl=self.nl, feat_dim=self.feat_dim, last_nl=None, layernormalize=self.layernormalize)

        if x_has_timesteps:
            x = unflatten_first_dim(x, sh)
        print("Shape before reshape: {}".format(x.get_shape().as_list()))
        x = tf.reshape(x, [-1, sh[1], self.feat_dim])
        return x

    def get_ac_value_nlp_extra_input(self, ob, vel, prev_ac, prev_rew):
        feed_ac = np.expand_dims(np.array(prev_ac), axis=1)
        #feed_ac = prev_ac
        feed_vel = np.expand_dims(np.array(vel), axis=1)
        feed_rew = np.expand_dims(np.array(prev_rew), axis=1)
        #feed_rew = prev_rew
        #print("Ac, vel, rew shapes:", feed_ac.shape, feed_vel.shape, feed_rew.shape)
        feed_dict = {self.ph_ob: ob[:, None], self.ph_vel: feed_vel, self.ph_prev_ac: feed_ac, self.ph_prev_rew: feed_rew, self.c_in_1: self.lstm1_c, self.h_in_1: self.lstm1_h}
        if self.lstm2_size > 0:
            feed_dict.update({self.c_in_2: self.lstm2_c, self.h_in_2: self.lstm2_h})
        if self.lstm2_size > 0:  
            a, vpred, nlp, self.lstm1_c, self.lstm1_h, self.lstm2_c, self.lstm2_h = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1, \
                                self.c_out_2, self.h_out_2],
                                feed_dict=feed_dict)
        else:
            a, vpred, nlp, self.lstm1_c, self.lstm1_h = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1], feed_dict=feed_dict)
        #print("LSTM1 c: {}".format(self.lstm1_c))
        #print("LSTM1 h: {}".format(self.lstm1_h))
        #print("LSTM2 c: {}".format(self.lstm2_c))
        #print("LSTM2 h: {}".format(self.lstm2_h))
        return a[:, 0], vpred[:, 0], nlp[:, 0]

    def get_ac_value_nlp(self, ob):
        feed_dict = {self.ph_ob: ob[:, None], self.c_in_1: self.lstm1_c, self.h_in_1: self.lstm1_h}
        if self.lstm2_size > 0:
            feed_dict.update({self.c_in_2: self.lstm2_c, self.h_in_2: self.lstm2_h})
        if self.lstm2_size > 0:  
            a, vpred, nlp, self.lstm1_c, self.lstm1_h, self.lstm2_c, self.lstm2_h = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1, \
                                self.c_out_2, self.h_out_2],
                                feed_dict=feed_dict)
        else:
            a, vpred, nlp, self.lstm1_c, self.lstm1_h = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1], feed_dict=feed_dict)
        #print("LSTM1 c: {}".format(self.lstm1_c))
        #print("LSTM1 h: {}".format(self.lstm1_h))
        #print("LSTM2 c: {}".format(self.lstm2_c))
        #print("LSTM2 h: {}".format(self.lstm2_h))
        return a[:, 0], vpred[:, 0], nlp[:, 0]
    

    def get_ac_value_nlp_eval(self, ob):
        feed_dict = {self.ph_ob: ((ob,),), self.c_in_1: self.lstm1_c_eval, self.h_in_1: self.lstm1_h_eval}
        if self.lstm2_size:
            feed_dict.update({self.c_in_2: self.lstm2_c_eval, self.h_in_2: self.lstm2_h_eval})
        if self.lstm2_size:  
            a, vpred, nlp, self.lstm1_c_eval, self.lstm1_h_eval, self.lstm2_c_eval, self.lstm2_h_eval = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1, \
                                self.c_out_2, self.h_out_2],
                                feed_dict=feed_dict)
        else:
            a, vpred, nlp, self.lstm1_c_eval, self.lstm1_h_eval = \
                getsess().run([self.a_samp, self.vpred, self.nlp_samp, self.c_out_1, self.h_out_1], feed_dict=feed_dict)
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
