from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import glob
import cv2
import sys
""" Input to this graph are going to be images of size 48x48 pixels which we are going to use to implement emotion recognition
    Our emotions are going to be: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral"""

def emotion_recognition_fn(features, labels, mode):

        # Input Layer
        input_layer = tf.reshape (features['x'], [-1, 48,48,1])
        input_layer = tf.cast(input_layer, tf.float32)
        # Output shape : [batch_size, 48, 48, 1] with regards to [batch_size, image_width, image_height, channels]
        # First convolutional layer - parameters need to be tuned
        conv1 = tf.layers.conv2d(
                                inputs = input_layer,
                                filters = 32,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        # Output shape should be: [batch_size, 48, 48, 64 ]
        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 24, 24, 64]
        conv2 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 64,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 12, 12, 64]
        conv3 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 128,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        # Output shape should be: [batch_size, 12, 12, 128 ]
        pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 6, 6, 128]
        pool3_flat  = tf.reshape(pool3, [-1, 12 * 12 * 128])
        # pool3_flat shape : [1,4608] - [batch_size, features]
        dense = tf.layers.dense(inputs = pool3_flat, units = 2048, activation = tf.nn.relu)
        dropout = tf.nn.dropout(dense, keep_prob = 0.7)
		# Output shape : [batch_size, 1024(hidden units)]
        logits = tf.layers.dense(inputs = dropout, units = 7)
        # We define output units to be 7(seven), corresponding to each of our emotions
        # Output shape should be [batch_size, 7]

        predictions = {
            "classes" : tf.argmax(input = logits, axis = 1),
            "probabilities" : tf.nn.softmax(logits, name = 'softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions['classes'])

        # Calculate Loss
        one_hot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 7)
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
            train_op = optimizer.minimize(
                loss = loss,
                global_step = tf.train.get_global_step()
            )
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op = train_op)


        eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def train_and_eval(img_folder, model_path):

	mypath = img_folder
	images = [cv2.imread(file) for file in sorted(glob.glob(mypath))]
	ims = []
	for image in images:
		ims.append(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
	images = np.asarray(ims)
	mean_image = np.mean(images, axis=0)
	images = np.subtract(images,mean_image)
	preds = []

	test_input_fn = tf.estimator.inputs.numpy_input_fn (
		x = {"x" : images},
		num_epochs = 1,
		shuffle = False,
		num_threads = 1 )

	fer2013_classifier = tf.estimator.Estimator(
		model_fn = emotion_recognition_fn,
		model_dir = model_path
		)
	pred_results = fer2013_classifier.predict(input_fn = test_input_fn)
	for prediction in pred_results:
		preds.append(prediction)
	print (preds)
	return preds

i_f = sys.argv[1]
m_p = sys.argv[2]
train_and_eval(i_f, m_p)
