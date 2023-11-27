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

        # get the tensorflow session
        config = tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=True)
        sess = tf.Session(config=config)

        # initialize all variables
        sess.run(tf.initialize_all_variables())

        feed_dict = {a: [2.0, 3.0], b: [3.0, 4.0]}

        # now run the sum operation
        ppx = sess.run([c], feed_dict)

        # print the result
        print(ppx)
        self.assertAllEqual(ppx[0], [5, 7.], description)
    return testAddNormal

if __name__ == '__main__':
    testsmap = {'float32': tf.float32}

    for name, param in testsmap.items():
        test_func = createAddTestFunction(name, param)
        setattr(AddTest, 'test_{0}'.format(name), test_func)

    tf.test.main()
