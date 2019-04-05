
import tensorflow as tf
from model import Model
import numpy as np
from common.shared_functions import glorot_variance, make_tf_variable


class SpatialRepresentation(Model):
    shape = None
    alpha = None

    def __init__(self, shape, settings, features, next_component=None):
        # next_component -> graph_representations.Representation(triples, encoder_settings), encoder_settings
        Model.__init__(self, next_component, settings)
        self.shape = shape  # feature_shape = [int(encoder_settings['EntityCount']), int(encoder_settings['FeatureDimension'])]
        self.features = features.astype(np.float32)

    def local_initialize_train(self):
        variance = glorot_variance(self.shape)
        self.alpha = make_tf_variable(0, variance, self.shape)

    def get_all_subject_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha)

    def get_all_object_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha)

    def get_all_codes(self, mode='train'):
        return tf.pow(self.features, self.alpha), None, tf.pow(self.features, self.alpha)

    def print(self):
        print('layer type:', type(self))