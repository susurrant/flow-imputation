import tensorflow as tf
from model import Model

class BilinearDiag(Model):
    X = None
    Y = None

    encoder_cache = {'train': None, 'test': None}

    def parse_settings(self):
        self.regularization_parameter = float(self.settings['RegularizationParameter'])

    def compute_codes(self, mode='train'):
        if self.encoder_cache[mode] is not None:
            return self.encoder_cache[mode]

        subject_codes, relation_codes, object_codes = self.next_component.get_all_codes(mode=mode)
        e1s = tf.nn.embedding_lookup(subject_codes, self.X[:, 0])
        rs = tf.nn.embedding_lookup(relation_codes, self.X[:, 1])
        e2s = tf.nn.embedding_lookup(object_codes, self.X[:, 2])

        self.encoder_cache[mode] = (e1s, rs, e2s)
        return self.encoder_cache[mode]


    def get_loss(self, mode='train'):
        e1s, rs, e2s = self.compute_codes(mode=mode)
        # It would be best to add a bias parameter because there is a gap between flow intensity 0 and the minimum intensity.
        energies = tf.reduce_sum(e1s * rs * e2s, 1) 
        weight = 1
        #return tf.reduce_mean(tf.losses.absolute_difference(self.Y, energies, weight))  # change loss function
        return tf.losses.mean_squared_error(self.Y, energies, weight)


    def local_initialize_train(self):
        self.X = tf.placeholder(tf.int32, shape=[None, 3], name='BiDiag_X')
        self.Y = tf.placeholder(tf.float32, shape=[None], name='BiDiag_Y')

    def local_get_train_input_variables(self):
        return [self.X, self.Y]

    def local_get_test_input_variables(self):
        return [self.X]

    def predict(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        energies = tf.reduce_sum(e1s * rs * e2s, 1) # sum by row
        return tf.nn.relu(energies)

    def predict_all_subject_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_subject_codes = self.next_component.get_all_subject_codes(mode='test')
        all_energies = tf.transpose(tf.matmul(all_subject_codes, tf.transpose(rs * e2s)))
        return tf.nn.relu(all_energies)

    def predict_all_object_scores(self):
        e1s, rs, e2s = self.compute_codes(mode='test')
        all_object_codes = self.next_component.get_all_object_codes(mode='test')
        all_energies = tf.matmul(e1s * rs, tf.transpose(all_object_codes))
        return tf.nn.relu(all_energies)

    def local_get_regularization(self):
        e1s, rs, e2s = self.compute_codes(mode='train')
        regularization = tf.reduce_mean(tf.square(e1s))
        regularization += tf.reduce_mean(tf.square(rs))
        regularization += tf.reduce_mean(tf.square(e2s))

        return self.regularization_parameter * regularization
