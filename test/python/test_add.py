import tensorflow as tf
import os
import tf_backend

class AddTest(tf.test.TestCase):
    longMessage = True

def createAddTestFunction(description, type):
    def testAddNormal(self):

        a = tf.placeholder(type, shape=[2])
        b = tf.placeholder(type, shape=[2])
        # build the sum operation
        c = a+b
        d = tf.placeholder(type, shape=[2])
        e = tf.placeholder(type, shape=[2])
        f = d+e
        g = c+f
        h = g+f
        i = tf.abs(h)
        j = -i
        # get the tensorflow session
        config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)
        sess = tf.Session(config=config)

        # initialize all variables

        feed_dict = {a: [2.0, 3.0], b: [3.0, 4.0],
                        d: [4.0, 5.0], e: [5.0, 6.0]}

        # now run the sum operation
        sess.run(tf.initialize_all_variables())
        ppx = sess.run([j], feed_dict)

        # print the result
        print(ppx)
        self.assertAllEqual(ppx[0], [-23., -29.], description)
    return testAddNormal

if __name__ == '__main__':
    testsmap = {'float32': tf.float32}

    for name, param in testsmap.items():
        test_func = createAddTestFunction(name, param)
        setattr(AddTest, 'test_{0}'.format(name), test_func)

    tf.test.main()
