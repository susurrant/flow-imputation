
from model import Model
import numpy as np


class SpatialRepresentation(Model):
    shape = None

    def __init__(self, shape, settings, features, next_component=None):
        # next_component -> graph_representations.Representation(triples, encoder_settings), encoder_settings
        Model.__init__(self, next_component, settings)
        self.shape = shape  # feature_shape = [int(encoder_settings['EntityCount']), int(encoder_settings['FeatureDimension'])]
        self.features = features.astype(np.float32)

    def get_all_subject_codes(self, mode='train'):
        return self.features

    def get_all_object_codes(self, mode='train'):
        return self.features

    def get_all_codes(self, mode='train'):
        return self.features, None, self.features

    def print(self):
        print('layer type:', type(self))