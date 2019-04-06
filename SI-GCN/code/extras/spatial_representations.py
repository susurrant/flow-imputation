
import tensorflow as tf
from model import Model
import numpy as np


class SpatialRepresentation(Model):
    shape = None
    alpha = None

    def __init__(self, shape, settings, features, next_component=None):
        # next_component -> graph_representations.Representation(triples, encoder_settings), encoder_settings
        Model.__init__(self, next_component, settings)
        self.shape = shape  # feature_shape = [int(encoder_settings['EntityCount']), int(encoder_settings['FeatureDimension'])]
        self.features = features.astype(np.float32)

    def local_initialize_train(self):
        co = np.ones((self.shape[0], 2)).astype(np.float32)
        feat = np.random.random(size=(self.shape[0], self.shape[1]-2)).astype(np.float32)
        self.alpha = tf.Variable(np.concatenate((co, feat), axis=1))

    def local_get_weights(self):
        return [self.alpha]

    def get_all_subject_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha)

    def get_all_object_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha)

    def get_all_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha), None, tf.pow(self.features, self.alpha)

    def print(self):
        print('layer type:', type(self))