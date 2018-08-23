# A simple python program to make inference on a model developped with tensorflow


import tensorflow as tf
import matplotlib.image as img
import numpy as np


graph = tf.get_default_graph()
sess = tf.Session()

# We launch a Session
with graph.as_default():
    with sess.as_default(): 
        saver = tf.train.import_meta_graph('test_network.cpkt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('./'))
        sess.run(tf.global_variables_initializer())
        image = img.imread('../4.pgm')
        test = np.expand_dims(image, axis=0)
        print 'Size of image =', np.shape(image)
        print 'Size of array =', np.shape(test)
        test_features = np.reshape(test, (1, 784))
        print 'Size of reshapedarray =', np.shape(test_features)
        # compute the predicted output for test_x
        x_tensor = graph.get_tensor_by_name('X:0')
        y_tensor = graph.get_tensor_by_name('Softmax/Softmax:0')
        pred_y = sess.run(y_tensor, feed_dict={x_tensor: test_features} )
        print(pred_y)