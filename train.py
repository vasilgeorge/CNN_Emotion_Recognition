from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import load_data
import emotion_recognition_model

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
        model_fn = emotion_recognition_model.emotion_recognition_fn,
        model_dir = '/Users/vasilgeorge/Documents/Imperial Machine Learning MSc 2017:18/Machine Learning/assignment2_advanced/CNN_Emotion_Recognition/public/ckpt'
    )

    # Set up logging for predictions
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LogginTensorHook(tensors = tensors_to_log, every_n_iter=50)
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
