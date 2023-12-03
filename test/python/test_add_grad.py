
import tensorflow as tf
import os
import tf_backend

class AddTest(tf.test.TestCase):
    longMessage = True

def createAddTestFunction(description, type):
    def testAddNormal(self):

        weights = tf.Variable(initial_value=tf.random_normal([3, 2]), trainable=True, name='weights')
        biases = tf.Variable(initial_value=tf.zeros([2, 2]), trainable=True, name='biases')


        dot_input = tf.placeholder(type, shape=[2, 3])

        # 定义矩阵乘法操作
        matrix_product = tf.matmul(dot_input, weights) + biases

        # 定义损失函数（这里用矩阵乘法的结果的平方和）
        loss = tf.reduce_sum(tf.square(matrix_product - tf.constant([[5.0, 6.0], [7.0, 8.0]], dtype=tf.float32)))

        # 定义优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

        # 定义反向传播操作
        trainable_vars = tf.trainable_variables()
        if not trainable_vars:
            raise ValueError("No trainable variables found.")

        # Optimization operation
        train_op = optimizer.minimize(loss, var_list=trainable_vars)


        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        sess = tf.Session(config=config)

        feed_dict = {dot_input: [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], weights: [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], biases : [[5.0, 6.0], [7.0, 8.0]]}
        # now run the sum operation
        sess.run(tf.initialize_all_variables())
        _, result = sess.run([train_op, matrix_product], feed_dict)
        print("Result of matrix multiplication:")
        print(result)

        print(ppx)
    return testAddNormal

if __name__ == '__main__':

    test_func = createAddTestFunction("simple", tf.float32)
    setattr(AddTest, 'test_{0}'.format("simple"), test_func)

    tf.test.main()
