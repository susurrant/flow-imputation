
import argparse
import sys, shutil, os, time
sys.path.append('./optimization')

import tensorflow as tf
from optimization.optimize import build_tensorflow
from common import settings_reader, io, model_builder, optimizer_parameter_parser, evaluation, auxilliaries
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on a given dataset.")
    parser.add_argument("--settings", help="Filepath for settings file.", required=True)
    parser.add_argument("--dataset", help="Filepath for dataset.", required=True)
    args = parser.parse_args()

    settings = settings_reader.read(args.settings)
    #print(settings)

    '''
    1. Load datasets
    '''
    dataset = args.dataset

    relations_path = dataset + '/relations.dict'
    entities_path = dataset + '/entities.dict'
    train_path = dataset + '/train.txt'
    valid_path = dataset + '/valid.txt'
    if settings['Evaluation']['Dataset'] == 'Negative':
        test_path = dataset + '/test_n.txt'
    elif settings['Evaluation']['Dataset'] == 'Positive':
        test_path = dataset + '/test.txt'
    feature_path = dataset + '/features.txt'

    train_triplets = io.read_triplets_as_list(train_path, entities_path, relations_path)
    valid_triplets = io.read_triplets_as_list(valid_path, entities_path, relations_path)
    test_triplets = io.read_triplets_as_list(test_path, entities_path, relations_path)
    features = io.read_features_as_list(feature_path)

    train_triplets = np.array(train_triplets)
    threshold = np.min(train_triplets[:, 3])
    # It would be best to add a bias parameter when calculating energy scores (in class BilinearDiag) rather than subtract the min intensity here.
    train_triplets[:, 3] -= threshold
    valid_triplets = np.array(valid_triplets)
    valid_triplets[:, 3] -= threshold
    test_triplets = np.array(test_triplets)
    features = np.array(features)

    entities = io.read_dictionary(entities_path)
    relations = io.read_dictionary(relations_path)

    '''
    2. Load general settings
    '''
    encoder_settings = settings['Encoder']
    decoder_settings = settings['Decoder']
    shared_settings = settings['Shared']
    general_settings = settings['General']
    optimizer_settings = settings['Optimizer']
    evaluation_settings = settings['Evaluation']

    general_settings.put('EntityCount', len(entities))
    general_settings.put('RelationCount', len(relations))
    general_settings.put('FeatureCount', len(features[0]))
    general_settings.put('EdgeCount', len(train_triplets))
    general_settings.put('GraphBatchSize', train_triplets.shape[0])
    print('Graph batch size:', train_triplets.shape[0])

    encoder_settings.merge(shared_settings)
    encoder_settings.merge(general_settings)
    decoder_settings.merge(shared_settings)
    decoder_settings.merge(general_settings)

    optimizer_settings.merge(general_settings)
    evaluation_settings.merge(general_settings)

    if decoder_settings['Output']:
        shutil.rmtree('../data/output')
        os.mkdir('../data/output')

    '''
    3. Construct the encoder-decoder pair:
    '''
    encoder = model_builder.build_encoder(encoder_settings, train_triplets, features)
    model = model_builder.build_decoder(encoder, decoder_settings)
    #print(encoder.needs_graph())

    '''
    4. Construct the optimizer with validation MRR as early stopping metric:
    '''
    opp = optimizer_parameter_parser.Parser(optimizer_settings)
    opp.set_save_function(model.save)

    scorer = evaluation.Scorer(evaluation_settings, model, threshold)

    def score_validation_data(validation_data):
        score_summary = scorer.compute_scores(validation_data, output=False).get_summary()
        #score_summary.pretty_print()

        lookup_string = score_summary.accuracy_string()
        early_stopping = score_summary.results[lookup_string]
        # True: output estimations
        score_summary = scorer.compute_scores(test_triplets, output=decoder_settings['Output']).get_summary()
        score_summary.pretty_print()

        return early_stopping

    opp.set_early_stopping_score_function(score_validation_data)

    adj_list = [[] for _ in entities]
    for i,triplet in enumerate(train_triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]

    def sample_edge_neighborhood(triplets, sample_size):

        edges = np.zeros(sample_size, dtype=np.int32)

        #initialize
        sample_counts = np.array([d for d in degrees])
        picked = np.array([False for _ in triplets])
        seen = np.array([False for _ in degrees])

        for i in range(0, sample_size):
            weights = sample_counts * seen

            if np.sum(weights) == 0:
                weights = np.ones_like(weights)
                weights[np.where(sample_counts == 0)] = 0

            probabilities = weights / np.sum(weights)
            chosen_vertex = np.random.choice(np.arange(degrees.shape[0]), p=probabilities)
            chosen_adj_list = adj_list[chosen_vertex]
            seen[chosen_vertex] = True

            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

            while picked[edge_number]:
                chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
                chosen_edge = chosen_adj_list[chosen_edge]
                edge_number = chosen_edge[0]

            edges[i] = edge_number
            other_vertex = chosen_edge[1]
            picked[edge_number] = True
            sample_counts[chosen_vertex] -= 1
            sample_counts[other_vertex] -= 1
            seen[other_vertex] = True

        return edges

    if 'NegativeSampleRate' in general_settings:
        ns = auxilliaries.NegativeSampler(float(general_settings['NegativeSampleRate']), general_settings['EntityCount'],
                                          entities.values())
        ns.set_positives(train_triplets)

        def t_func(x):
            arr = np.array(x)
            if not encoder.needs_graph():
                return ns.transform(arr)
            else:
                graph_batch_size = int(general_settings['GraphBatchSize'])
                graph_batch_ids = sample_edge_neighborhood(arr, graph_batch_size)
                graph_batch = train_triplets[graph_batch_ids]

                # Apply dropouts:
                graph_percentage = float(general_settings['GraphSplitSize'])
                split_size = int(graph_percentage * graph_batch.shape[0])
                graph_split_ids = np.random.choice(graph_batch_ids, size=split_size, replace=False)
                graph_split = train_triplets[graph_split_ids]

                samples = ns.transform_exclusive(graph_batch)
                return graph_split[:, :3], samples[0], samples[1]
                #return (graph_split[:, :3], graph_batch[:, :3], graph_batch[:, 3])

        opp.set_sample_transform_function(t_func)


    '''
    5. Initialize for training:
    '''
    # Hack for validation evaluation:
    model.preprocess(train_triplets)
    model.register_for_test(train_triplets)

    model.initialize_train()

    optimizer_weights = model.get_weights()
    optimizer_input = model.get_train_input_variables()
    loss = model.get_loss(mode='train') + model.get_regularization()


    '''
    6. Add additiontional ops:
    '''
    for add_op in model.get_additional_ops():
        opp.additional_ops.append(add_op)

    optimizer_parameters = opp.get_parametrization()

    print('---------------optimizer weights---------------')
    print(optimizer_weights)
    print('-----------------------------------------------')
    print('----------------optimizer input----------------')
    print(optimizer_input)  #[<tf.Tensor 'graph_edges:0' shape=(?, 3) dtype=int32>,
                            # <tf.Tensor 'BiDiag_X:0' shape=(?, 3) dtype=int32>,
                            # <tf.Tensor 'BiDiag_Y:0' shape=(?,) dtype=float32>]
    print('---------------optimizer weights---------------')
    print('--------------optimizer parameters-------------')
    print(optimizer_parameters)
    print('-----------------------------------------------')


    '''
    7. Train with Converge:
    '''
    start_time = time.clock()
    # when trainning the model, SampleTransformer in optimization generates triplets and labels to feed the model
    model.session = tf.Session()
    print('\nbuild tensorflow...')
    optimizer = build_tensorflow(loss, optimizer_weights, optimizer_parameters, optimizer_input)
    optimizer.set_session(model.session)
    print('\nfit...')
    optimizer.fit(train_triplets, validation_data=valid_triplets)
    print('Total running time: %.2f' % ((time.clock()-start_time)/60.0), 'mins')
