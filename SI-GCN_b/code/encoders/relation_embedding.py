import numpy as np
import tensorflow as tf
from model import Model

class RelationEmbedding(Model):
    shape=None

    def __init__(self, shape, settings, next_component=None):
        Model.__init__(self, next_component, settings) # next_component = BasisGcn
        self.shape = shape   # relation_shape = [int(encoder_settings['EntityCount']), int(encoder_settings['CodeDimension'])]

    def parse_settings(self):
        self.embedding_width = int(self.settings['CodeDimension'])

    def local_initialize_train(self):
        relation_initial = np.random.randn(self.shape[0], self.shape[1]).astype(np.float32)

        self.W_relation = tf.Variable(relation_initial)

    def local_get_weights(self):
        return [self.W_relation]

    def get_all_codes(self, mode='train'):
        codes = self.next_component.get_all_codes(mode=mode)
        print('---------------------------------------------------------------')
        print('RelationEmbedding -> get_all_codes')
        print('  codes[0]', codes[0].get_shape())
        print('  self.W_relation', self.W_relation.get_shape())
        print('  codes[2]', codes[2].get_shape())
        print('---------------------------------------------------------------')
        #print('-------RelationEmbedding.get_all_codes', codes[0], self.W_relation, codes[2])
        return codes[0], self.W_relation, codes[2]