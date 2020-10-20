import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import h5py
import torch
from collections import defaultdict
from model import PixelCNN


class TF2Pytorch(object):
    V = 'V.ExponentialMovingAverage'
    g = 'g.ExponentialMovingAverage'
    b = 'b.ExponentialMovingAverage'

    def __init__(self, tf_weights):
        self.state_dict = {}
        self.layer_count = defaultdict(int)
        self.tf_weights = tf_weights

    def load_tf_tensor(self, prefix, var):
        name = '.'.join([prefix, var])
        return torch.from_numpy(self.tf_weights[name][:])

    def load_conv2d(self, name_torch, name_tf):
        self.state_dict[name_torch + '.weight_v'] = self.load_tf_tensor(name_tf, self.V).permute(3, 2, 0, 1)
        self.state_dict[name_torch + '.weight_g'] = self.load_tf_tensor(name_tf, self.g)[:, None, None, None]
        self.state_dict[name_torch + '.bias'] = self.load_tf_tensor(name_tf, self.b).squeeze()
        self.layer_count['conv2d'] += 1

    def load_deconv2d(self, name_torch, name_tf):
        self.state_dict[name_torch + '.weight_v'] = self.load_tf_tensor(name_tf, self.V).permute(3, 2, 0, 1)
        self.state_dict[name_torch + '.weight_g'] = self.load_tf_tensor(name_tf, self.g)[:, None, None, None]
        self.state_dict[name_torch + '.bias'] = self.load_tf_tensor(name_tf, self.b).squeeze()
        self.layer_count['deconv2d'] += 1

    def load_dense(self, name_torch, name_tf):
        self.state_dict[name_torch + '.weight_v'] = self.load_tf_tensor(name_tf, self.V).permute(1, 0)
        self.state_dict[name_torch + '.weight_g'] = self.load_tf_tensor(name_tf, self.g)[:, None]
        self.state_dict[name_torch + '.bias'] = self.load_tf_tensor(name_tf, self.b)
        self.layer_count['dense'] += 1

    def load_pixelcnn(self):
        torch_to_tf_dict = {}
        layer_count = {}
        up_nr_resnet = [5, 5, 5]
        down_nr_resnet = [5, 6, 6]

        TF_ROOT = 'model'

        # ////// input  //////
        conv_count = self.layer_count['conv2d']
        tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
        self.load_conv2d('u_init.conv', tf_conv2d_name)
        conv_count = self.layer_count['conv2d']
        tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
        self.load_conv2d('ul_init.0.conv', tf_conv2d_name)
        conv_count = self.layer_count['conv2d']
        tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
        self.load_conv2d('ul_init.1.conv', tf_conv2d_name)

        # ////// up     //////
        torch_name = 'up_layers'

        for block_id, nr_resnet in enumerate(up_nr_resnet):
            layer_name = '.'.join([torch_name, str(block_id)])

            for j in range(nr_resnet):
                # ////// u_stream //////
                u_stream_prefix = '.'.join([layer_name, 'u_stream', str(j)])

                torch_conv2d_name = '.'.join([u_stream_prefix, 'conv_input', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                torch_conv2d_name = '.'.join([u_stream_prefix, 'conv_out', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                # ////// ul stream //////
                ul_stream_prefix = '.'.join([layer_name, 'ul_stream', str(j)])

                torch_conv2d_name = '.'.join([ul_stream_prefix, 'conv_input', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                torch_dense_name = '.'.join([ul_stream_prefix, 'nin_skip', 'lin_a'])
                dense_count = self.layer_count['dense']
                tf_dense_name = '.'.join([TF_ROOT, f'dense_{dense_count}'])
                self.load_dense(torch_dense_name, tf_dense_name)

                torch_conv2d_name = '.'.join([ul_stream_prefix, 'conv_out', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

            if block_id != 2:
                torch_conv2d_name = '.'.join(['downsize_u_stream', str(block_id), 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                torch_conv2d_name = '.'.join(['downsize_ul_stream', str(block_id), 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

        # ////// down   //////
        torch_name = 'down_layers'

        for block_id, nr_resnet in enumerate(down_nr_resnet):
            layer_name = '.'.join([torch_name, str(block_id)])

            for j in range(nr_resnet):
                # ////// u_stream //////
                u_stream_prefix = '.'.join([layer_name, 'u_stream', str(j)])

                torch_conv2d_name = '.'.join([u_stream_prefix, 'conv_input', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                torch_dense_name = '.'.join([u_stream_prefix, 'nin_skip', 'lin_a'])
                dense_count = self.layer_count['dense']
                tf_dense_name = '.'.join([TF_ROOT, f'dense_{dense_count}'])
                self.load_dense(torch_dense_name, tf_dense_name)

                torch_conv2d_name = '.'.join([u_stream_prefix, 'conv_out', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                # ////// ul stream //////
                ul_stream_prefix = '.'.join([layer_name, 'ul_stream', str(j)])

                torch_conv2d_name = '.'.join([ul_stream_prefix, 'conv_input', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

                torch_dense_name = '.'.join([ul_stream_prefix, 'nin_skip', 'lin_a'])
                dense_count = self.layer_count['dense']
                tf_dense_name = '.'.join([TF_ROOT, f'dense_{dense_count}'])
                self.load_dense(torch_dense_name, tf_dense_name)

                torch_conv2d_name = '.'.join([ul_stream_prefix, 'conv_out', 'conv'])
                conv_count = self.layer_count['conv2d']
                tf_conv2d_name = '.'.join([TF_ROOT, f'conv2d_{conv_count}'])
                self.load_conv2d(torch_conv2d_name, tf_conv2d_name)

            if block_id != 2:
                torch_deconv2d_name = '.'.join(['upsize_u_stream', str(block_id), 'deconv'])
                deconv_count = self.layer_count['deconv2d']
                tf_deconv2d_name = '.'.join([TF_ROOT, f'deconv2d_{deconv_count}'])
                self.load_deconv2d(torch_deconv2d_name, tf_deconv2d_name)

                torch_deconv2d_name = '.'.join(['upsize_ul_stream', str(block_id), 'deconv'])
                deconv_count = self.layer_count['deconv2d']
                tf_deconv2d_name = '.'.join([TF_ROOT, f'deconv2d_{deconv_count}'])
                self.load_deconv2d(torch_deconv2d_name, tf_deconv2d_name)

        # ////// output //////
        dense_count = self.layer_count['dense']
        tf_dense_name = '.'.join([TF_ROOT, f'dense_{dense_count}'])
        self.load_dense('nin_out.lin_a', tf_dense_name)


tf_path = './tf_model/params_cifar.ckpt'
hdf5_path = './tf_model/tf_weights.h5'
ckpt_path = './tf_model/pixelcnn.pth'

init_vars = tf.train.list_variables(tf_path)

if os.path.isfile(hdf5_path):
    h5f = h5py.File(hdf5_path, 'r')
else:
    with h5py.File(hdf5_path, 'w') as h5f:
        for name, shape in init_vars:

            val = tf.train.load_variable(tf_path, name)

            print(val.dtype)
            print("Loading TF weight {} with shape {}, {}".format(name, shape, val.shape))
            torch.from_numpy(np.array(val))
            if 'model' in name:
                new_name = name.replace('/', '.')
                print(new_name)
                h5f.create_dataset(str(new_name), data=val)

    h5f = h5py.File(hdf5_path, 'r')


model = PixelCNN(nr_resnet=5, nr_filters=160,
                 input_channels=3, nr_logistic_mix=10)

#print(model.state_dict().keys())
converter = TF2Pytorch(h5f)
converter.load_pixelcnn()

model.load_state_dict(converter.state_dict)
torch.save(model.state_dict(), ckpt_path)
h5f.close()

