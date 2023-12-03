import tensorflow as tf
import tf_backend

# 创建两个输入变量
x = tf.Variable([2.0, 3.0], name='x', trainable=True)
y = tf.Variable([1.0, 2.0], name='y', trainable=True)

# 定义点积操作
dot_product = tf.reduce_sum(tf.multiply(x, y), name='dot_product')

# 定义损失函数（这里用点积的平方）
loss = tf.square(dot_product, name='loss')

# 定义优化器
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 定义反向传播操作
train_op = optimizer.minimize(loss)

# 创建 TensorFlow 会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 执行前向传播和反向传播操作
    _, dot_product_value, loss_value = sess.run([train_op, dot_product, loss])

    # 输出结果
    print("Dot product:", dot_product_value)
    print("Loss:", loss_value)