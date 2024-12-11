# # models/model_zinc.py
# import copy
# from keras import backend as K
# from keras import objectives
# from keras.models import Model
# from keras.layers import Input, Dense, Lambda
# from keras.layers.core import Dense, Activation, Flatten, RepeatVector
# from keras.layers.wrappers import TimeDistributed
# from keras.layers.recurrent import GRU
# from keras.layers.convolutional import Convolution1D
# import tensorflow as tf
# import zinc_grammar as G

# # helper variables in Keras format for parsing the grammar
# masks_K      = K.variable(G.masks)
# ind_of_ind_K = K.variable(G.ind_of_ind)

# MAX_LEN = 277
# DIM = G.D


# class MoleculeVAE():

#     autoencoder = None

#     def create(self,
#                charset,
#                max_length = MAX_LEN,
#                latent_rep_size = 2,
#                weights_file = None):
#         charset_length = len(charset)

#         x = Input(shape=(max_length, charset_length))
#         _, z = self._buildEncoder(x, latent_rep_size, max_length)
#         self.encoder = Model(x, z)

#         encoded_input = Input(shape=(latent_rep_size,))
#         self.decoder = Model(
#             encoded_input,
#             self._buildDecoder(
#                 encoded_input,
#                 latent_rep_size,
#                 max_length,
#                 charset_length
#             )
#         )

#         x1 = Input(shape=(max_length, charset_length))
#         vae_loss, z1 = self._buildEncoder(x1, latent_rep_size, max_length)
#         self.autoencoder = Model(
#             x1,
#             self._buildDecoder(
#                 z1,
#                 latent_rep_size,
#                 max_length,
#                 charset_length
#             )
#         )

#         # for obtaining mean and log variance of encoding distribution
#         x2 = Input(shape=(max_length, charset_length))
#         (z_m, z_l_v) = self._encoderMeanVar(x2, latent_rep_size, max_length)
#         self.encoderMV = Model(input=x2, output=[z_m, z_l_v])

#         if weights_file:
#             self.autoencoder.load_weights(weights_file)
#             self.encoder.load_weights(weights_file, by_name = True)
#             self.decoder.load_weights(weights_file, by_name = True)
#             self.encoderMV.load_weights(weights_file, by_name = True)

#         self.autoencoder.compile(optimizer = 'Adam',
#                                  loss = vae_loss,
#                                  metrics = ['accuracy'])


#     def _encoderMeanVar(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
#         h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
#         h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
#         h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
#         h = Flatten(name='flatten_1')(h)
#         h = Dense(435, activation = 'relu', name='dense_1')(h)

#         z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
#         z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

#         return (z_mean, z_log_var)


#     def _buildEncoder(self, x, latent_rep_size, max_length, epsilon_std = 0.01):
#         h = Convolution1D(9, 9, activation = 'relu', name='conv_1')(x)
#         h = Convolution1D(9, 9, activation = 'relu', name='conv_2')(h)
#         h = Convolution1D(10, 11, activation = 'relu', name='conv_3')(h)
#         h = Flatten(name='flatten_1')(h)
#         h = Dense(435, activation = 'relu', name='dense_1')(h)

#         def sampling(args):
#             z_mean_, z_log_var_ = args
#             batch_size = K.shape(z_mean_)[0]
#             epsilon = K.random_normal(shape=(batch_size, latent_rep_size), mean=0., std = epsilon_std)
#             return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

#         z_mean = Dense(latent_rep_size, name='z_mean', activation = 'linear')(h)
#         z_log_var = Dense(latent_rep_size, name='z_log_var', activation = 'linear')(h)

#         # this function is the main change.
#         # essentially we mask the training data so that we are only allowed to apply
#         #   future rules based on the current non-terminal
#         def conditional(x_true, x_pred):
#             most_likely = K.argmax(x_true)
#             most_likely = tf.reshape(most_likely,[-1]) # flatten most_likely
#             ix2 = tf.expand_dims(tf.gather(ind_of_ind_K, most_likely),1) # index ind_of_ind with res
#             ix2 = tf.cast(ix2, tf.int32) # cast indices as ints
#             M2 = tf.gather_nd(masks_K, ix2) # get slices of masks_K with indices
#             M3 = tf.reshape(M2, [-1,MAX_LEN,DIM]) # reshape them
#             P2 = tf.mul(K.exp(x_pred),M3) # apply them to the exp-predictions
#             P2 = tf.div(P2,K.sum(P2,axis=-1,keepdims=True)) # normalize predictions
#             return P2

#         def vae_loss(x, x_decoded_mean):
#             x_decoded_mean = conditional(x, x_decoded_mean) # we add this new function to the loss
#             x = K.flatten(x)
#             x_decoded_mean = K.flatten(x_decoded_mean)
#             xent_loss = max_length * objectives.binary_crossentropy(x, x_decoded_mean)
#             kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis = -1)
#             return xent_loss + kl_loss

#         return (vae_loss, Lambda(sampling, output_shape=(latent_rep_size,), name='lambda')([z_mean, z_log_var]))

#     def _buildDecoder(self, z, latent_rep_size, max_length, charset_length):
#         h = Dense(latent_rep_size, name='latent_input', activation = 'relu')(z)
#         h = RepeatVector(max_length, name='repeat_vector')(h)
#         h = GRU(501, return_sequences = True, name='gru_1')(h)
#         h = GRU(501, return_sequences = True, name='gru_2')(h)
#         h = GRU(501, return_sequences = True, name='gru_3')(h)
#         return TimeDistributed(Dense(charset_length), name='decoded_mean')(h) # don't do softmax, we do this in the loss now

#     def save(self, filename):
#         self.autoencoder.save_weights(filename)

#     def load(self, charset, weights_file, latent_rep_size = 2, max_length=MAX_LEN):
#         self.create(charset, max_length = max_length, weights_file = weights_file, latent_rep_size = latent_rep_size)


# models/model_zinc_pytorch.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import zinc_grammar as G

MAX_LEN = 277
DIM = G.D


class MoleculeVAE(nn.Module):
    def __init__(
        self, charset, max_length=MAX_LEN, latent_rep_size=56, epsilon_std=0.01
    ):
        super(MoleculeVAE, self).__init__()
        self.charset = charset
        self.max_length = max_length
        self.latent_rep_size = latent_rep_size
        self.epsilon_std = epsilon_std
        self.charset_length = len(charset)

        # 注册masks和ind_of_ind为buffer
        self.register_buffer("masks", torch.tensor(G.masks, dtype=torch.float32))
        self.register_buffer("ind_of_ind", torch.tensor(G.ind_of_ind, dtype=torch.long))

        # 对应Keras层命名和结构:
        # Encoder部分
        self.conv_1 = nn.Conv1d(
            in_channels=self.charset_length, out_channels=9, kernel_size=9
        )
        self.conv_2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv_3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)
        # flatten_1 对应 flatten操作，不是独立层，在forward中实现
        # dense_1
        # 卷积输出长度计算: (参见原说明)
        # after conv1: 277-9+1=269
        # after conv2: 269-9+1=261
        # after conv3: 261-11+1=251
        conv_out_dim = 10 * 251
        self.dense_1 = nn.Linear(conv_out_dim, 435)

        # z_mean, z_log_var
        self.z_mean = nn.Linear(435, latent_rep_size)
        self.z_log_var = nn.Linear(435, latent_rep_size)

        # Decoder部分
        self.latent_input = nn.Linear(latent_rep_size, latent_rep_size)
        # repeat_vector 不存在独立层，在forward中用repeat实现
        self.gru_1 = nn.GRU(
            input_size=latent_rep_size, hidden_size=501, batch_first=True
        )
        self.gru_2 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.gru_3 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.decoded_mean = nn.Linear(501, self.charset_length)

    def encode(self, x):
        # x: (batch, max_length, charset_length)
        # 转置为 (batch, charset_length, max_length)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))
        # flatten_1
        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.dense_1(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean) * self.epsilon_std
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z

    def decode(self, z):
        h = F.relu(self.latent_input(z))
        # repeat_vector
        h = h.unsqueeze(1).repeat(1, self.max_length, 1)
        h, _ = self.gru_1(h)
        h, _ = self.gru_2(h)
        h, _ = self.gru_3(h)
        # decoded_mean
        out = self.decoded_mean(h)
        return out

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded_logits = self.decode(z)
        return x_decoded_logits, z_mean, z_log_var

    def conditional(self, x_true, x_pred):
        # 还原原始conditional逻辑
        batch = x_true.size(0)
        max_length = x_true.size(1)
        most_likely = torch.argmax(x_true, dim=-1)  # (batch, max_length)
        most_likely = most_likely.view(-1)  # (batch*max_length)
        ix2 = self.ind_of_ind[most_likely].unsqueeze(1)
        M2 = self.masks[ix2.squeeze(1), :]  # (batch*max_length, DIM)
        M3 = M2.view(batch, max_length, DIM)
        P2 = torch.exp(x_pred) * M3
        P2 = P2 / (P2.sum(dim=-1, keepdim=True) + 1e-8)
        return P2

    def vae_loss(self, x, x_decoded_logits, z_mean, z_log_var):
        # 与原Keras版本对应的loss
        x_decoded_mean = self.conditional(x, x_decoded_logits)
        x_flat = x.view(-1)
        x_decoded_flat = x_decoded_mean.view(-1)
        recon_loss = (
            F.binary_cross_entropy(x_decoded_flat, x_flat, reduction="mean")
            * self.max_length
        )
        kl_loss = -0.5 * torch.mean(
            1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var)
        )
        return recon_loss + kl_loss, recon_loss, kl_loss


class WrapperMoleculeVAE(object):
    # 为了保持与原始代码中MoleculeVAE类的接口一致（.create, .load），提供一个外层封装
    def __init__(self):
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.encoderMV = None
        self.model = None

    def create(self, charset, max_length=MAX_LEN, latent_rep_size=2, weights_file=None):
        # 创建PyTorch模型
        self.model = MoleculeVAE(
            charset, max_length=max_length, latent_rep_size=latent_rep_size
        )

        # 无法直接使用原Keras的load_weights，对于完全一致的权重兼容很难
        # 这里如果提供weights_file是Keras的hdf5文件，就无法直接加载
        # 如需加载torch权重可以torch.load()

        # 与Keras接口保持一致：encoder, decoder, encoderMV
        # encoder: 输入x -> z
        # decoder: 输入z -> x_decoded
        # encoderMV: 输入x -> (z_mean, z_log_var)
        # 在Pytorch中可用model内部方法替代，但这里为接口一致性，不实际分拆模型。
        # 实际使用中，可以在forward中实现encoder、decoder逻辑，也可单独写出子module。
        self.autoencoder = self.model
        # 暂不完全拆分encoder和decoder为独立model，但保持接口占位

        # 优化器与compile等在train中完成，这里不调用。

    def load(self, charset, weights_file, latent_rep_size=2, max_length=MAX_LEN):
        self.create(charset, max_length=max_length, latent_rep_size=latent_rep_size)
        # 无法直接加载Keras权重，但接口保留
        # 若有相应torch权重可执行:
        # self.model.load_state_dict(torch.load(weights_file))
        pass

    def save(self, filename):
        # 保存当前模型的state_dict
        torch.save(self.model.state_dict(), filename)
