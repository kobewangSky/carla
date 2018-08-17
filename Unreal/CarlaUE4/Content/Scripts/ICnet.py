import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI

import os
import sys
import timeit
import numpy as np
from scipy import misc

from Model import ICNet_BN
from tools import decode_labels

import cv2
import numpy as np


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)
# define setting & model configuration
ADE20k_class = 150  # predict: [0~149] corresponding to label [1~150], ignore class 0 (background)
cityscapes_class = 19

model_paths = {'train': '/icnet_cityscapes_train_30k.npy',
               'trainval': '/icnet_cityscapes_trainval_90k.npy',
               'train_bn': '/icnet_cityscapes_train_30k_bnnomerge.npy',
               'trainval_bn': '/icnet_cityscapes_trainval_90k_bnnomerge.npy'}

SAVE_DIR = '/output/'

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def load_img(img_path):
    if os.path.isfile(img_path):
        print('successful load img: {0}'.format(img_path))
    else:
        print('not found file: {0}'.format(img_path))
        sys.exit(0)

    filename = img_path.split('/')[-1]
    img = misc.imread(img_path, mode='RGB')
    print('input image shape: ', img.shape)

    return img, filename


def preprocess(img):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN

    img = tf.expand_dims(img, dim=0)

    return img


def check_input(img):
    ori_h, ori_w = img.get_shape().as_list()[1:3]

    if ori_h % 32 != 0 or ori_w % 32 != 0:
        new_h = (int(ori_h / 32) + 1) * 32
        new_w = (int(ori_w / 32) + 1) * 32
        shape = [new_h, new_w]

        img = tf.image.pad_to_bounding_box(img, 0, 0, new_h, new_w)

        print('Image shape cannot divided by 32, padding to ({0}, {1})'.format(new_h, new_w))
    else:
        shape = [ori_h, ori_w]

    return img, shape



class ICnet(TFPluginAPI):

	#expected api: setup your model for your use cases
	def onSetup(self):
		#setup or load your model and pass it into stored
		
		#Usually store session, graph, and model if using keras
		self.scripts_path = ue.get_content_dir() + "Scripts"

		self.model_directory = self.scripts_path + "/model/ICNET"
		print("self.model_directory =" + self.scripts_path + "--------------------------------------------------")
		modeltype = 'trainval_bn'
		num_classes = cityscapes_class
		imageShape = (1024, 2048, 3)
		self.x = tf.placeholder(dtype=tf.float32, shape=imageShape)
		img_tf = preprocess(self.x)
		img_tf, n_shape = check_input(img_tf)


		model = ICNet_BN
		self.net = model({'data': img_tf}, num_classes=num_classes, filter_scale=1)
		raw_output = self.net.layers['conv6_cls']
		raw_output_up = tf.image.resize_bilinear(raw_output, size=n_shape, align_corners=True)

		raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, imageShape[0], imageShape[1])
		raw_output_up = tf.argmax(raw_output_up, axis=3)
		self.pred = decode_labels(raw_output_up, imageShape, num_classes)
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config=config)
		init = tf.global_variables_initializer()

		self.sess.run(init)
		model_path = self.model_directory + model_paths[modeltype]
		
		print("model_path =" + model_path + "--------------------------------------------------")

		self.net.load(model_path, self.sess)
		print('Restore from {}'.format(model_path))
	
	#expected api: storedModel and session, json inputs
	def onJsonInput(self, jsonInput):
		#e.g. our json input could be a pixel array
		#pixelarray = jsonInput['pixels']

		#run input on your graph
		#e.g. sess.run(model['y'], feed_dict)
		# where y is your result graph and feed_dict is {x:[input]}

		#...

		#you can also call an event e.g.
		#callEvent('myEvent', 'myData')

		#return a json you will parse e.g. a prediction
		pixelarray = jsonInput['pixels']

		x_raw = np.reshape(pixelarray, (1024, 2048, 4))
		#print(x_raw)
		print("------------------------------------------------------------------")
		bgr = x_raw[...,[2,1,0]]
		cv2.imshow("image", bgr)

		#_, filename = load_img(self.scripts_path + '/input/Test2.png')
		#print(filename)

		start_time = timeit.default_timer()
		preds = self.sess.run(self.pred, feed_dict={self.x: bgr})
		elapsed = timeit.default_timer() - start_time
		print('inference time: {}'.format(elapsed))
		
		misc.imsave(self.scripts_path + SAVE_DIR + "Test2.png", preds[0])

		result = {}
		result['prediction'] = -1

		return result

	#optional api: no params forwarded for training? TBC
	def onBeginTraining(self):
		print("onBeginTraining ICnet----------------------------------")
		#train here

		#...

		#inside your training loop check if we should stop early
		#if(this.shouldStop):
		#	break
		result = {}
		result['elapsed'] = "onBeginTrainingOK-------------------"
		return 

	#optional api: use if you need some things to happen if we get stopped
	def onStopTraining(self):
		#you should be listening to this.shouldstop, but you can also receive this call
		pass

#required function to get our api
def getApi():
	#return CLASSNAME.getInstance()
	return ICnet.getInstance()