
import tensorflow_backend.algorithms as tensorflow_algorithms
import shared.algorithms as shared_algorithms
from abstract import BaseOptimizer
import tensorflow as tf

'''
Partial class for shared architecture between tensorflow optimizers.
'''

class Optimizer:

    def __init__(self, stack):
        self.stack = stack

    def fit(self, training_data, validation_data=None):
        self.stack.set_training_data(training_data)
        if validation_data is not None:
            self.stack.set_validation_data(validation_data)

        self.initialize_for_fitting()

        i = 0
        next_batch = self.stack.next_batch()
        while next_batch is not None:
            i+=1
            self.stack.set_iteration(i)

            processed_batch = self.stack.process_data(next_batch)
            train_loss = self.update_from_batch(processed_batch)
            
            if self.stack.postprocess(train_loss) == 'stop':
                print("Stopping training.")
                break

            next_batch = self.stack.next_batch()
        
class TensorflowOptimizer(Optimizer):

    def set_placeholders(self, placeholders):
        self.placeholders = placeholders

    def set_session(self, session):
        self.session = session
        
    def compute_functions(self, loss_function, parameters_to_optimize):
        self.loss_function = self.stack.process_loss_function(loss_function)
        self.gradient_function = self.stack.process_gradient_function(self.loss_function, parameters_to_optimize)
        self.update_function = self.stack.process_update_function(self.gradient_function, parameters_to_optimize)
        self.variables = parameters_to_optimize
        
    def loss(self, placeholder_input):
        #self.session = tf.Session()
        placeholder_input = self.stack.process_data(placeholder_input)
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)
        
        feed_dict = dict(zip(self.placeholders, placeholder_input))
        return self.session.run(self.loss_function, feed_dict=feed_dict) 
    
    def gradients(self, placeholder_input):
        #self.session = tf.Session()

        placeholder_input = self.stack.process_data(placeholder_input)
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)
        
        feed_dict = dict(zip(self.placeholders, placeholder_input))
        return self.session.run(self.gradient_function, feed_dict=feed_dict)

    def initialize_for_fitting(self):
        #self.session = tf.Session()
        self.stack.set_session(self.session)
        
        init_op = tf.initialize_all_variables()
        self.session.run(init_op)

    def update_from_batch(self, processed_batch):        
        feed_dict = dict(zip(self.placeholders, processed_batch))

        adds = self.stack.get_additional_ops()
        upd = self.session.run([self.update_function, self.loss_function, adds], feed_dict=feed_dict)

        return upd[1]


def __from_component(component_name):
    print('    ', component_name)
    if component_name == "GradientDescent":
        return tensorflow_algorithms.GradientDescent
    
    if component_name == "Minibatches":
        return shared_algorithms.Minibatches

    if component_name == "IterationCounter":
        return shared_algorithms.IterationCounter

    if component_name == "SampleTransformer":
        return shared_algorithms.SampleTransformer

    if component_name == "GradientClipping":
        return tensorflow_algorithms.GradientClipping

    if component_name == "EarlyStopper":
        return shared_algorithms.EarlyStopper
        
    if component_name == "AdaGrad":
        return tensorflow_algorithms.AdaGrad

    if component_name == "RmsProp":
        return tensorflow_algorithms.RmsProp

    if component_name == "Adam":
        return tensorflow_algorithms.Adam

    if component_name == "ModelSaver":
        return shared_algorithms.ModelSaver

    if component_name == "TrainLossReporter":
        return shared_algorithms.TrainLossReporter

    if component_name == "AdditionalOp":
        return tensorflow_algorithms.AdditionalOp
        
    
def __construct_optimizer(settings):
    optimizer = BaseOptimizer()
    print('optimizer components:')
    for component, parameters in settings:  # recursive construction
        optimizer = __from_component(component)(optimizer, parameters)

    #TODO: Better error handling
    if not optimizer.verify():
        print("Construction failed.")


    return TensorflowOptimizer(optimizer)

def build_tensorflow(loss_function, parameters_to_optimize, settings, placeholders):
    optimizer = __construct_optimizer(settings)
    
    optimizer.compute_functions(loss_function, parameters_to_optimize)
    optimizer.set_placeholders(placeholders)
    
    return optimizer
    
