import cv2
import tensorflow as tf
import numpy as np
import os
import h5py
import random
from PIL import Image

class Net:
  def __init__(self, vgg16_batch_size=40, train_batch_size=200, test_batch_size=30, learning_rate=1e-1, epsilon=1e-6):
    seed = 1234
    self.vgg16_batch_size = vgg16_batch_size
    self.train_batch_size = train_batch_size
    self.test_batch_size = test_batch_size
    self.learning_rate = learning_rate
    self.variance_epsilon = 1e-6
    self.epsilon = epsilon
    self.weight_list = {}
    tf.set_random_seed(seed)
    np.random.seed(seed)

  def create_weights(self, shape, stddev = 0.01):
    if len(shape)==4:
      layer_name = 'conv' + str(len(self.weight_list))
    else:
      layer_name = 'fc' + str(len(self.weight_list))
    
    weights = tf.Variable(tf.random_normal(shape, stddev=stddev))
    self.weight_list[layer_name] = weights
    return weights
      
  def create_biases(self, shape, stddev=0.01):
    layer_name = 'bias' + str(len(self.weight_list))
    biases = tf.Variable(tf.random_normal(shape, stddev=stddev))
    self.weight_list[layer_name] = biases
    return biases

      
  def create_conv_layer(self, input, weights, biases=None , activation=None, padding='SAME'):
    layer = tf.nn.conv2d(input=input,
              filter=weights,
              strides=[1,1,1,1],
              padding=padding)
    if not biases is None:
      layer = tf.add( layer,
              biases)
    if activation == 'relu':
      layer = tf.nn.relu(layer)
    elif activation == 'tanh':
      layer = tf.nn.tanh(layer)
    elif activation == 'sigmoid':
      layer = tf.nn.sigmoid(layer)
    return layer

  def create_pool_layer(self, input):
    layer = tf.nn.max_pool(  value = input,
                ksize = [1,2,2,1],
                strides = [1,2,2,1],
                padding = 'SAME')
    return layer

  def create_avg_pool_layer(self, input):
    layer = tf.nn.avg_pool(  value = input,
                ksize = [1,2,2,1],
                strides = [1,2,2,1],
                padding = 'SAME')
    return layer

  def create_upsample_layer(self, input, weights, output_size=None, strides=(2,2), biases=None, activation=None):
    input_shape = input.get_shape()
    output_shape = None
    if output_size is None:
      output_shape = [tf.shape(input)[0], input_shape[1].value*2, input_shape[2].value*2, weights.get_shape()[2].value]
    else:
      output_shape = [tf.shape(input)[0], output_size[0], output_size[1], weights.get_shape()[2].value]
    
    layer = tf.nn.conv2d_transpose(
              value=input,
              filter=weights,
              output_shape=output_shape,
              strides=[1,strides[0],strides[1],1])
    if not biases is None:
      layer = tf.add(layer, biases)
    if activation=='relu':
      layer = tf.nn.relu(layer)
    elif activation == 'tanh':
      layer = tf.nn.tanh(layer)
    elif activation == 'sigmoid':
      layer = tf.nn.sigmoid(layer)
    return layer

  def create_fc_layer(self, input, weights, biases = None, activation=None):
    layer = tf.matmul(a=input, b=weights)
    if not biases is None:
      layer = tf.add(layer, biases)
    if activation=='relu':
      layer = tf.nn.relu(layer)
    elif activation == 'tanh':
      layer = tf.nn.tanh(layer)
    elif activation == 'sigmoid':
      layer = tf.nn.sigmoid(layer)
    return layer

  def preprocess_input(self, images):
    dests = images - np.array([123.68, 116.779, 103.939], dtype=np.float32)
    temp = np.copy(dests[:,:,:,0])
    dests[:,:,:,0] = dests[:,:,:,2]
    dests[:,:,:,2] = temp
    return dests

  def write_weights(self, session, weight_file='weight.hdf5', write_mean_variance=False):
    f = h5py.File(weight_file, 'w')
    for key in self.weight_list.keys():
      if not write_mean_variance:
        if not key.startswith('mean') and not key.startswith('variance'):
          f.create_dataset(key, data = session.run(self.weight_list[key]))
      else:
        f.create_dataset(key, data = session.run(self.weight_list[key]))
    f.close()

  def compute_vgg16_block_outputs(self, input, vgg16_weight_file='vgg16_notop.h5'):
    f = h5py.File(vgg16_weight_file, 'r')
    # block1
    vgg16_output = self.create_conv_layer(
                      input=input,
                      weights=tf.Variable(np.float32(f['block1_conv1']['block1_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block1_conv1']['block1_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block1_conv2']['block1_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block1_conv2']['block1_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)

    vgg16_block1_output = vgg16_output
    # block2
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block2_conv1']['block2_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block2_conv1']['block2_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block2_conv2']['block2_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block2_conv2']['block2_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)

    vgg16_block2_output = vgg16_output
    # block3
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv1']['block3_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv1']['block3_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv2']['block3_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv2']['block3_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv3']['block3_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv3']['block3_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)

    vgg16_block3_output = vgg16_output
    # block4
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv1']['block4_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv1']['block4_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv2']['block4_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv2']['block4_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv3']['block4_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv3']['block4_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)

    vgg16_block4_output = vgg16_output
    # block5
    '''
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv1']['block5_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv1']['block5_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv2']['block5_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv2']['block5_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv3']['block5_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv3']['block5_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)

    vgg16_block5_output = vgg16_output
    '''
    f.close()
    return vgg16_block1_output, vgg16_block2_output, vgg16_block3_output, vgg16_block4_output#, vgg16_block5_output

  def compute_vgg16_output(self, input, vgg16_weight_file='vgg16_notop.h5'):
    f = h5py.File(vgg16_weight_file, 'r')
    # block1
    vgg16_output = self.create_conv_layer(
                      input=input,
                      weights=tf.Variable(np.float32(f['block1_conv1']['block1_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block1_conv1']['block1_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block1_conv2']['block1_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block1_conv2']['block1_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)
    # block2
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block2_conv1']['block2_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block2_conv1']['block2_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block2_conv2']['block2_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block2_conv2']['block2_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)
    # block3
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv1']['block3_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv1']['block3_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv2']['block3_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv2']['block3_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block3_conv3']['block3_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block3_conv3']['block3_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)
    # block4
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv1']['block4_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv1']['block4_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv2']['block4_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv2']['block4_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block4_conv3']['block4_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block4_conv3']['block4_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)
    # block5
    '''
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv1']['block5_conv1_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv1']['block5_conv1_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv2']['block5_conv2_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv2']['block5_conv2_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_conv_layer(input=vgg16_output,
                      weights=tf.Variable(np.float32(f['block5_conv3']['block5_conv3_W_1:0']), trainable=False),
                      biases=tf.Variable(np.float32(f['block5_conv3']['block5_conv3_b_1:0']), trainable=False),
                      activation='relu')
    vgg16_output = self.create_pool_layer(vgg16_output)
    '''
    f.close()
    return vgg16_output

  # ごま塩ノイズの追加
  def add_salt_pepper_noise(self, image, strong=1000):
    height, width = image.shape[:2]
    noise_image = image.copy()
    white_pix_x = np.random.randint(0, width -1, strong)
    white_pix_y = np.random.randint(0, height -1, strong)
    noise_image[(white_pix_y, white_pix_x)] = (255, 255, 255)
    black_pix_x = np.random.randint(0, width -1, strong)
    black_pix_y = np.random.randint(0, height -1, strong)
    noise_image[(black_pix_y, black_pix_x)] = (0, 0, 0)
    return noise_image
  
  # ガウシアンノイズの追加
  def add_gaussian_noise(self, image, mean=0, sigma=15):
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma,(row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    gauss_image = image + gauss
    return gauss_image
  
  def add_gaussian_blur(self, image, kernel_size=5):
    blur_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blur_image
  
  # アルファチャンネルを意識した画像の合成
  # overlay_image is a PNG image opened by 
  # cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2BGRA)
  def blend(self, src_image, overlay_image):
    width, height = src_image.shape[:2]

    dst = src_image.copy()
    src = overlay_image.copy()

    mask = src[:,:,3] # アルファチャンネルだけ抜き出す。
    mask = mask / 255.0 # 0-255だと使い勝手が悪いので、0.0-1.0に変更。

    #src = src[:,:,:3] # アルファチャンネルは取り出しちゃったのでもういらない。
    dst = np.array(dst, dtype=np.float32)
    src = np.array(src, dtype=np.float32)
    dst[:, :, 0] *= 1 - mask # 透過率に応じて元の画像を暗くする。
    dst[:, :, 1] *= 1 - mask
    dst[:, :, 2] *= 1 - mask
    dst[:, :, 0] += src[:, :, 0] * mask # 貼り付ける方の画像に透過率をかけて加算。
    dst[:, :, 1] += src[:, :, 1] * mask
    dst[:, :, 2] += src[:, :, 2] * mask
    return dst

  def compute_mean_variance_parameters(self, precomputed_file='data.h5', weight_file='weight.h5'):
    PX = tf.placeholder(dtype=tf.float32, shape=[None, self.feature_size, self.feature_size, 512])
    PY = self.load_model(PX, weight_file)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    keys = []
    params = []
    for key in self.weight_list:
      if key.startswith('mean') or key.startswith('variance'):
        keys.append(key)
        params.append(self.weight_list[key])
    
    f = h5py.File(precomputed_file, 'r')
    dset_input = f['input']
    num_data = dset_input.len()
    weights = {}
    for i in range(0, num_data, self.train_batch_size):
      print('Progress', i, '/', num_data)
      end_i = min(num_data, i+self.train_batch_size)
      inputs = dset_input[i:end_i]
      if i!=end_i:
        inputs=np.array(inputs)
        datas = session.run(params, feed_dict={PX: inputs})
      for j in range(len(keys)):
        key = keys[j]
        if i==0:
          weights[key] = datas[j]*len(inputs)
        else:
          weights[key] += datas[j]*len(inputs)
    f.close()
    f = h5py.File(weight_file, 'a')
    for key in keys:
      weights[key] = weights[key] / num_data
      # Save to file
      f.create_dataset(key, data=np.float32(weights[key]))    
    f.close()

class StyleMixer(Net):
  def __init__(self, img_width=256, img_height=256,img_channel=3,num_epochs=20,train_batch_size=20, test_batch_size=20, vgg16_batch_size=20, learning_rate=1e-1, epsilon=1e-6, variance_epsilon=1e-6):
    Net.__init__(self, train_batch_size=train_batch_size, test_batch_size=test_batch_size, learning_rate=learning_rate),
    self.img_width = img_width
    self.img_height = img_height
    self.img_channel = img_channel
    self.num_epochs = num_epochs
    self.epsilon = epsilon
    self.variance_epsilon = variance_epsilon

  def create_dataset(self, content_folder='./content/', style_folder='./style/'):
    content_images = os.listdir(content_folder)
    content_images.sort()
    contents = []
    for content_image in content_images:
      contents.append(content_folder + content_image)

    style_images = os.listdir(style_folder)
    style_images.sort()
    styles = []
    for style_image in style_images:
      styles.append(style_folder + style_image)

    return {'style': styles, 'content': contents}

  def compute_instance_normalization_params(self, layer):
    m = tf.reduce_mean(layer, axis=[1,2], keepdims=True)
    v = layer - m
    v = tf.sqrt(tf.reduce_mean(tf.square(v), axis=[1,2], keepdims=True) + self.variance_epsilon)
    return m, v

  def load_model(self, content_holder, style_holders, weight_file='weight.style_transfer.h5'):
    f = h5py.File(weight_file, 'r')
    #content_mean, content_variance = self.compute_instance_normalization_params(content_holder)
    #style_mean, style_variance = self.compute_instance_normalization_params(style_holders[3])
    # synthesize features of content and style
    #syn_feature_layer0 = tf.multiply(style_variance, tf.divide(content_holder - content_mean, content_variance))+style_mean
    
    # decoder
    # block 1
    # layer 1
    weights = tf.Variable(np.float32(f['conv0']))
    self.weight_list['conv0'] = weights
    biases = tf.Variable(np.float32(f['bias1']))
    self.weight_list['bias1'] = biases
    syn_feature_layer1 = self.create_upsample_layer(
            content_holder,
            weights=weights,
            biases=biases,
            output_size=(style_holders[2].get_shape()[1].value, style_holders[2].get_shape()[2].value),
            activation='relu')
    # layer 2
    weights = tf.Variable(np.float32(f['conv2']))
    self.weight_list['conv2'] = weights
    biases = tf.Variable(np.float32(f['bias3']))
    self.weight_list['bias3'] = biases
    syn_feature_layer1 = self.create_conv_layer(
            syn_feature_layer1,
            weights=weights,
            biases=biases,
            activation='relu')

    # layer 3
    weights = tf.Variable(np.float32(f['conv4']))
    self.weight_list['conv4'] = weights
    biases = tf.Variable(np.float32(f['bias5']))
    self.weight_list['bias5'] = biases
    syn_feature_layer1 = self.create_conv_layer(
            syn_feature_layer1,
            weights=weights,
            biases=biases,
            activation='relu')

    # block 2 
    # layer 1
    weights = tf.Variable(np.float32(f['conv6']))
    self.weight_list['conv6'] = weights
    biases = tf.Variable(np.float32(f['bias7']))
    self.weight_list['bias7'] = biases
    syn_feature_layer2 = self.create_upsample_layer(
            syn_feature_layer1,
            weights=weights,
            biases=biases,
            output_size=(style_holders[1].get_shape()[1].value, style_holders[1].get_shape()[2].value),
            activation='relu')

    # layer 2
    weights = tf.Variable(np.float32(f['conv8']))
    self.weight_list['conv8'] = weights
    biases = tf.Variable(np.float32(f['bias9']))
    self.weight_list['bias9'] = biases
    syn_feature_layer2 = self.create_conv_layer(
            syn_feature_layer2,
            weights=weights,
            biases=biases,
            activation='relu')

    # block 3
    # layer 1
    weights = tf.Variable(np.float32(f['conv10']))
    self.weight_list['conv10'] = weights
    biases = tf.Variable(np.float32(f['bias11']))
    self.weight_list['bias11'] = biases
    syn_feature_layer3 = self.create_upsample_layer(
            syn_feature_layer2,
            weights=weights,
            biases=biases,
            output_size=(style_holders[0].get_shape()[1].value, style_holders[0].get_shape()[2].value),
            activation='relu')

    # layer 2
    weights = tf.Variable(np.float32(f['conv12']))
    self.weight_list['conv12'] = weights
    biases = tf.Variable(np.float32(f['bias13']))
    self.weight_list['bias13'] = biases
    syn_feature_layer3 = self.create_conv_layer(
            syn_feature_layer3,
            weights=weights,
            biases=biases,
            activation='relu')

    # block 4
    # layer 1
    weights = tf.Variable(np.float32(f['conv14']))
    self.weight_list['conv14'] = weights
    biases = tf.Variable(np.float32(f['bias15']))
    self.weight_list['bias15'] = biases
    syn_img = self.create_upsample_layer(
            syn_feature_layer3,
            weights=weights,
            biases=biases,
            output_size=(self.img_height, self.img_width),
            activation='sigmoid')
    syn_img = syn_img * 255.0

    f.close()
    # image of type (B,G,R)
    return content_holder, syn_feature_layer1, syn_feature_layer2, syn_feature_layer3, syn_img

  def compute_gram_matrix(self, layer):
    layer_shape = layer.get_shape()
    mat = tf.reshape(layer, [-1, layer_shape[1].value*layer_shape[2].value, layer_shape[3].value])
    transpose_mat = tf.transpose(mat, [0,2,1])
    return tf.matmul(transpose_mat, mat)
    
  def compute_cost(self, image_features, content_feature, style_features, style_loss_weight=1.0):
    content_loss = tf.reduce_mean(tf.square(content_feature - image_features[3]))
    
    style_loss1 = tf.reduce_mean(tf.square(self.compute_gram_matrix(image_features[0]) - self.compute_gram_matrix(style_features[0])))
    style_loss2 = tf.reduce_mean(tf.square(self.compute_gram_matrix(image_features[1]) - self.compute_gram_matrix(style_features[1])))
    style_loss3 = tf.reduce_mean(tf.square(self.compute_gram_matrix(image_features[2]) - self.compute_gram_matrix(style_features[2])))
    style_loss4 = tf.reduce_mean(tf.square(self.compute_gram_matrix(image_features[3]) - self.compute_gram_matrix(style_features[3])))
  
    style_loss = (style_loss1 + style_loss2 + style_loss3 + style_loss4)* style_loss_weight
    loss = content_loss + style_loss 
    
    return loss
   
  def mix_style(self, content_img, style_img, output_img, vgg16_weight_file='vgg16_notop.h5'):
    I = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel])
    F = self.compute_vgg16_block_outputs(I, vgg16_weight_file=vgg16_weight_file)
    
    content = np.float32(Image.open(content_img))
    content = np.reshape(content, [-1, self.img_height, self.img_width, self.img_channel])
    content = self.preprocess_input(content)
    
    style = np.float32(Image.open(style_img))
    style = np.reshape(style, [-1, self.img_height, self.img_width, self.img_channel])
    style = self.preprocess_input(style)
    
    output = tf.Variable(content, dtype=tf.float32)
    output_features = self.compute_vgg16_block_outputs(output, vgg16_weight_file=vgg16_weight_file)
    
    F0_shape = F[0].get_shape()
    F1_shape = F[1].get_shape()
    F2_shape = F[2].get_shape()
    F3_shape = F[3].get_shape()
    content_feature_holder = tf.placeholder(tf.float32, [None, F3_shape[1].value, F3_shape[2].value, F3_shape[3].value])
    style_features_holder = [tf.placeholder(tf.float32, [None, F0_shape[1].value, F0_shape[2].value, F0_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F1_shape[1].value, F1_shape[2].value, F1_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F2_shape[1].value, F2_shape[2].value, F2_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F3_shape[1].value, F3_shape[2].value, F3_shape[3].value])]
    
    cost = self.compute_cost(output_features, content_feature_holder, style_features_holder, style_loss_weight=0.000001)
    optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost)
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    content_feature = session.run(F, feed_dict={I: content})
    content_feature = content_feature[3]
    
    style_features = session.run(F, feed_dict={I: style})
    
    for i in range(self.num_epochs):
      loss, _ = session.run([cost, optimizer], feed_dict={
        content_feature_holder: content_feature,
        style_features_holder[0]: style_features[0],
        style_features_holder[1]: style_features[1],
        style_features_holder[2]: style_features[2],
        style_features_holder[3]: style_features[3]})
      print('Epoch', i, 'Loss', loss)
    output = session.run(output)
    output = np.reshape(output, [self.img_height, self.img_width, self.img_channel])
    output = output + np.float32([103.939, 116.779, 123.68])
    cv2.imwrite(output_img, output)
    
  def mix_all(self, content_folder, style_folder, output_folder, vgg16_weight_file='vgg16_notop.h5'):
    content_files = os.listdir(content_folder)
    style_files = os.listdir(style_folder)
    
    I = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_channel])
    F = self.compute_vgg16_block_outputs(I, vgg16_weight_file=vgg16_weight_file)
    output_features = self.compute_vgg16_block_outputs(I, vgg16_weight_file=vgg16_weight_file)
    
    F0_shape = F[0].get_shape()
    F1_shape = F[1].get_shape()
    F2_shape = F[2].get_shape()
    F3_shape = F[3].get_shape()
    content_feature_holder = tf.placeholder(tf.float32, [None, F3_shape[1].value, F3_shape[2].value, F3_shape[3].value])
                      
    style_features_holder = [tf.placeholder(tf.float32, [None, F0_shape[1].value, F0_shape[2].value, F0_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F1_shape[1].value, F1_shape[2].value, F1_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F2_shape[1].value, F2_shape[2].value, F2_shape[3].value]),
                      tf.placeholder(tf.float32, [None, F3_shape[1].value, F3_shape[2].value, F3_shape[3].value])]
    
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    
    for content_file in content_files:
      content_images = np.float32([np.float32(Image.open(content_folder + content_file))])
      content_images = self.preprocess_input(content_images)
      content_features = session.run(F, feed_dict={I: content_images})
      for i in range(0, len(style_files), self.train_batch_size):
        style_images = []
        end_i = min(len(style_files), i+self.train_batch_size)
        for j in range(i,end_i):
          print('Mix', content_file, style_files[j])
          style_images.append(np.float32(Image.open(style_folder+style_files[j])))
        style_images = np.float32(style_images)
        style_images = self.preprocess_input(style_images)
        style_features = session.run(F, feed_dict={I: style_images}) 
        
        output = []
        for j in range(i, end_i):
          output.append(content_images[0])
        output = np.float32(output)
        output = tf.Variable(output)
        output_features = self.compute_vgg16_block_outputs(output)
        cost = self.compute_cost(output_features, content_feature_holder, style_features_holder, style_loss_weight=0.000001)
        optimizer = tf.train.AdadeltaOptimizer(self.learning_rate).minimize(cost)
        session.run(tf.global_variables_initializer())
        
        for k in range(self.num_epochs):
          loss, _ = session.run([cost, optimizer], feed_dict={
          content_feature_holder: content_features[3],
          style_features_holder[0]: style_features[0],
          style_features_holder[1]: style_features[1],
          style_features_holder[2]: style_features[2],
          style_features_holder[3]: style_features[3]})
          
          print('Epoch', k, 'Loss', loss)
        
        output = session.run(output)
        output = output + np.float32([103.939, 116.779, 123.68])
        for k in range(len(output)):
          print('Write file', output_folder + content_file[:-4] + '_' + style_files[i+k][:-4] + '.jpg')
          cv2.imwrite(output_folder + content_file[:-4] + '_' + style_files[i+k][:-4] + '.jpg', output[k])
    
model = StyleMixer(
  img_width=256,
  img_height=256,
  img_channel=3,
  train_batch_size=10,
  test_batch_size=10,
  num_epochs=200,
  learning_rate=1e+3,
  epsilon=1e-6,
  variance_epsilon=1e-6)

'''
model.mix_style(
  content_img = './Harun.jpg',
  style_img = './cur_style/style_31.jpg',
  output_img = './output/Harun_31.jpg',
  vgg16_weight_file='vgg16_notop.h5')
'''


model.mix_all(
  content_folder='./content/',
  style_folder='./style/',
  output_folder='./output/',
  vgg16_weight_file='vgg16_notop.h5')