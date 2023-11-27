import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tf_backend

# 下载MNIST数据集
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 定义网络参数
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

# 定义输入和输出
x = tf.placeholder(tf.float32, [None, 784])  # 输入为28x28=784维的向量
y = tf.placeholder(tf.float32, [None, 10])   # 输出为10类

# 定义权重和偏置
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# 定义模型
pred = tf.nn.softmax(tf.matmul(x, W) + b)

# 定义损失函数
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 初始化变量
init = tf.global_variables_initializer()

# 开始训练
with tf.Session() as sess:
    sess.run(init)

    # 训练循环
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        # 遍历所有的批次
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch

        # 每个epoch打印一次信息
        if (epoch + 1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Optimization Finished!")

    # 测试模型
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
