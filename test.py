import tensorflow as tf 

with tf.variable_scope("H"):
  
    h_input = tf.Variable(tf.zeros([5,5]), name='LatentSpaceData')
    h_list = tf.trainable_variables()
print(h_input)
