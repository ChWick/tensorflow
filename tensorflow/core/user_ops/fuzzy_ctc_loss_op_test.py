import tensorflow as tf

class FuzzyCTCLossText(tf.test.TestCase):
    def testFuzzyCTCLoss(self):
        fuzzy_ctc_loss_module = tf.load_op_library("./fuzzy_ctc_op.so")
        with self.test_session():
            print("test")

if __name__ == "__main__":
    tf.test.main()
