import tensorflow as tf

hello = tf.constant("Hello, world")

print(hello.numpy().decode('utf-8'))