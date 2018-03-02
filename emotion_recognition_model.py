from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import train
import load_data

tf.logging.set_verbosity(tf.logging.INFO)


""" Input to this graph are going to be images of size 48x48 pixels which we are going to use to implement emotion recognition
    Our emotions are going to be: 0=Angry, 1=Fear, 2=Happy, 3=Sad, 4=Disgust, 5=Surprise, 6=Neutral"""

def emotion_recognition_fn(features, labels, mode):

        print("Giving input...")
        # Input Layer
        input_layer = tf.reshape (features['x'], [-1, 48,48,1])
        input_layer = tf.cast(input_layer, tf.float32)
        print(input_layer)
        # Output shape : [batch_size, 48, 48, 1] with regards to [batch_size, image_width, image_height, channels]
        print("First Convolutional Layer...")
        # First convolutional layer - parameters need to be tuned
        conv1 = tf.layers.conv2d(
                                inputs = input_layer,
                                filters = 32,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        # Output shape should be: [batch_size, 48, 48, 32 ]
        print("First Pooling Layer...")
        print(conv1)
        pool1 = tf.layers.max_pooling2d(inputs = conv1, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 24, 24, 32]
        print(pool1)
        conv2 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 64,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        # Output shape should be: [batch_size, 24, 24, 64 ]
        print(conv2)
        pool2 = tf.layers.max_pooling2d(inputs = conv2, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 12, 12, 64]
        print(pool2)
        conv3 = tf.layers.conv2d(
                                inputs = pool1,
                                filters = 128,
                                kernel_size = [5,5],
                                padding = "same",
                                activation = tf.nn.relu
                                )
        # Output shape should be: [batch_size, 12, 12, 128 ]
        print(conv3)
        pool3 = tf.layers.max_pooling2d(inputs = conv3, pool_size = [2,2], strides = 2)
        # Output shape should be: [batch_size, 6, 6, 128]
        print(pool3)
        pool3_flat  = tf.reshape(pool3, [-1, 12 * 12 * 128])
        print(pool3_flat)
        # pool3_flat shape : [1,4608] - [batch_size, features]

        dense = tf.layers.dense(inputs = pool3_flat, units = 1024, activation = tf.nn.relu)
        print(dense)
        dropout = tf.nn.dropout(dense, keep_prob = 0.7)#, training_mode == tf.estimator.ModeKeys.TRAIN)
        print(dropout)
        # Output shape : [batch_size, 1024(hidden units)]

        logits = tf.layers.dense(inputs = dropout, units = 7)

        # We define output units to be 7(seven), corresponding to each of our emotions
        # Output shape should be [batch_size, 7]

        predictions = {
            "classes" : tf.argmax(input = logits, axis = 1),
            "probabilities" : tf.nn.softmax(logits, name = 'softmax_tensor')
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode = mode, predictions = predictions)

        # Calculate Loss
        one_hot_labels = tf.one_hot(indices = tf.cast(labels, tf.int32), depth = 7)
        print(one_hot_labels)
        print(logits)
        loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)
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


def train_and_eval():

    fer2013 = load_data.get_FER2013_data()
    print("Splitting data in training and evaluating...")
    train_data = fer2013['X_train']
    train_labels  = np.asarray(fer2013['y_train'], dtype = np.int32)
    eval_data = fer2013['X_val']
    eval_labels = np.asarray(fer2013['y_val'], dtype = np.int32)
    print("Splitted data...")
    # Create an estimator
    fer2013_classifier = tf.estimator.Estimator(
        model_fn = emotion_recognition_fn,
        model_dir = '/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/ckpt'
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter=50)
    print("Setting up training function...")
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    print("Starting training...")
    fer2013_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])
    print("Finished training, starting evaluation...")
        # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

    eval_results = fer2013_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

train_and_eval()
