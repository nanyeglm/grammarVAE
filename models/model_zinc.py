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






# model_zinc.py (PyTorch version with GPU-compatible indexing)

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

        # 将 grammar 中的 masks 和 ind_of_ind 注册为 buffer
        # 这样当 model.to(device) 时，会自动迁移到对应设备
        self.register_buffer("masks", torch.tensor(G.masks, dtype=torch.float32))
        self.register_buffer("ind_of_ind", torch.tensor(G.ind_of_ind, dtype=torch.long))

        # Encoder网络结构
        # 输入 (batch, max_length, charset_length) 转为 (batch, charset_length, max_length)
        self.conv1 = nn.Conv1d(
            in_channels=self.charset_length, out_channels=9, kernel_size=9
        )
        self.conv2 = nn.Conv1d(in_channels=9, out_channels=9, kernel_size=9)
        self.conv3 = nn.Conv1d(in_channels=9, out_channels=10, kernel_size=11)

        # 计算卷积输出维度:
        # after conv1: length = 277-9+1=269
        # after conv2: length = 269-9+1=261
        # after conv3: length = 261-11+1=251
        # total = 10*251=2510
        conv_out_dim = 10 * 251

        self.fc_hidden = nn.Linear(conv_out_dim, 435)
        self.z_mean = nn.Linear(435, latent_rep_size)
        self.z_log_var = nn.Linear(435, latent_rep_size)

        # Decoder网络结构
        self.decoder_input = nn.Linear(latent_rep_size, latent_rep_size)
        self.gru_1 = nn.GRU(
            input_size=latent_rep_size, hidden_size=501, batch_first=True
        )
        self.gru_2 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.gru_3 = nn.GRU(input_size=501, hidden_size=501, batch_first=True)
        self.fc_out = nn.Linear(501, self.charset_length)

    def encode(self, x):
        # x: (batch, max_length, charset_length)
        x = x.permute(0, 2, 1)  # 转换为 (batch, charset_length, max_length)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.contiguous().view(x.size(0), -1)
        x = F.relu(self.fc_hidden(x))
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        return z_mean, z_log_var

    def reparameterize(self, z_mean, z_log_var):
        eps = torch.randn_like(z_mean) * self.epsilon_std
        z = z_mean + torch.exp(z_log_var / 2) * eps
        return z

    def decode(self, z):
        z = F.relu(self.decoder_input(z))
        z = z.unsqueeze(1).repeat(1, self.max_length, 1)
        out, _ = self.gru_1(z)
        out, _ = self.gru_2(out)
        out, _ = self.gru_3(out)
        out = self.fc_out(out)  # (batch, max_length, charset_length)
        return out

    def forward(self, x):
        z_mean, z_log_var = self.encode(x)
        z = self.reparameterize(z_mean, z_log_var)
        x_decoded_logits = self.decode(z)
        return x_decoded_logits, z_mean, z_log_var

    def conditional(self, x_true, x_pred):
        # x_true, x_pred: (batch, max_length, charset_length)
        batch = x_true.size(0)
        max_length = x_true.size(1)

        most_likely = torch.argmax(x_true, dim=-1)  # (batch, max_length)
        most_likely = most_likely.view(-1)  # (batch*max_length)

        # 使用 ind_of_ind 索引，需确保 ind_of_ind 已在同一设备上
        ix2 = self.ind_of_ind[most_likely].unsqueeze(1)  # (batch*max_length,1)

        # 根据 ix2 从 masks 中 gather 相应行
        # ix2.squeeze(1)是 (batch*max_length,)
        M2 = self.masks[ix2.squeeze(1), :]  # (batch*max_length, D)

        # reshape 回 (batch, max_length, DIM)
        M3 = M2.view(batch, max_length, DIM)

        P2 = torch.exp(x_pred) * M3
        P2 = P2 / (P2.sum(dim=-1, keepdim=True) + 1e-8)
        return P2

    def vae_loss(self, x, x_decoded_logits, z_mean, z_log_var):
        # 应用conditional函数
        x_decoded_mean = self.conditional(x, x_decoded_logits)

        # flatten
        x_flat = x.view(-1)
        x_decoded_flat = x_decoded_mean.view(-1)

        # binary cross entropy
        recon_loss = (
            F.binary_cross_entropy(x_decoded_flat, x_flat, reduction="mean")
            * self.max_length
        )
        kl_loss = -0.5 * torch.mean(
            1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var)
        )
        return recon_loss + kl_loss, recon_loss, kl_loss
