import numpy as np
import random


class NegativeSampler:

    negative_sample_rate = None
    n_entities = None
    
    def __init__(self, negative_sample_rate, n_entities, gids):
        self.negative_sample_rate = negative_sample_rate
        self.n_entities = n_entities
        self.gids = list(gids)
        self.objs = {}
        self.subs = {}
        self.positives = None
        self.negatives = []

    def set_positives(self, triplets):
        self.positives = triplets

        for triplet in triplets:
            if triplet[0] not in self.objs:
                self.objs[triplet[0]] = []

            self.objs[triplet[0]].append((triplet[1], triplet[2]))

            if triplet[2] not in self.subs:
                self.subs[triplet[2]] = []

            self.subs[triplet[2]].append((triplet[1], triplet[0]))

    def set_negatives(self):
        p_dict = set(map(tuple, self.positives[:, [0, 2]]))
        n_dict = set()
        for i in range(len(self.gids)):  # can be optimized using set operations
            for j in range(len(self.gids)):
                if j != i:
                    n_dict.add((self.gids[i], self.gids[j]))
        n_dict -= p_dict
        for g in n_dict:
            self.negatives.append([g[0], 0, g[1], 0])  # 只考虑一种关系，第二项值为0

        self.negatives = np.array(self.negatives)
        np.random.shuffle(self.negatives)

    # based on train triplets
    def sample(self, triplets):
        size_of_batch = len(triplets)
        number_to_generate = size_of_batch * self.negative_sample_rate
        if number_to_generate > self.negatives.shape[0]:
            number_to_generate = self.negatives.shape[0]

        neg_samples = self.negatives[random.sample(range(self.negatives.shape[0]), int(number_to_generate))]

        return np.concatenate((triplets, neg_samples), axis=0)

    # based on batch triplets
    def transform(self, triplets):
        size_of_batch = len(triplets)
        number_to_generate = size_of_batch*self.negative_sample_rate
        
        new_labels = np.zeros((size_of_batch * (self.negative_sample_rate + 1))).astype(np.uint16)
        new_indexes = np.tile(triplets, (self.negative_sample_rate + 1,1)).astype(np.uint16)
        new_labels[:size_of_batch] = triplets[:, 3]

        choices = np.random.binomial(1, 0.5, number_to_generate)
        values = np.random.randint(self.n_entities, size=number_to_generate)

        for i in range(size_of_batch):
            for j in range(self.negative_sample_rate):
                index = i+j*size_of_batch

                if choices[index]:
                    new_indexes[index+size_of_batch,2] = values[index]
                else:
                    new_indexes[index+size_of_batch,0] = values[index]

        return new_indexes[:, :3], new_labels

    def transform_exclusive(self, triplets):
        size_of_batch = len(triplets)
        number_to_generate = size_of_batch * self.negative_sample_rate

        new_labels = np.zeros((size_of_batch * (self.negative_sample_rate + 1))).astype(np.uint16)
        new_indexes = np.tile(triplets, (self.negative_sample_rate + 1, 1)).astype(np.uint16)
        new_labels[:size_of_batch] = triplets[:, 3]

        choices = np.random.binomial(1, 0.5, number_to_generate)

        for i in range(size_of_batch):
            for j in range(self.negative_sample_rate):
                index = i + j * size_of_batch + size_of_batch

                if choices[index - size_of_batch]:
                    new_indexes[index, 2] = random.randint(0, self.n_entities-1)
                    while (new_indexes[index][1], new_indexes[index][2]) in self.objs[new_indexes[index][0]]:
                        new_indexes[index, 2] = random.randint(0, self.n_entities-1)
                else:
                    new_indexes[index, 0] = random.randint(0, self.n_entities-1)
                    while (new_indexes[index][1], new_indexes[index][0]) in self.subs[new_indexes[index][2]]:
                        new_indexes[index, 0] = random.randint(0, self.n_entities-1)

        return new_indexes[:, :3], new_labels


class RelationFilter:

    def __init__(self, n_keep):
        self.n_keep = n_keep - 1

    def register(self, triplets, original_relations):
        d = {k:0 for k in original_relations}

        for triplet in triplets:
            i = original_relations[triplet[1]]
            d[i] += 1

        tuples = sorted(d.items(), key=lambda x: x[1], reverse=True)
        kept_relations = [t[0] for t in tuples[:self.n_keep]]
        discarded_relations = [t[0] for t in tuples[self.n_keep:]]

        self.d = {}
        for v,k in enumerate(kept_relations):
            self.d[k] = v

        for k in discarded_relations:
            self.d[k] = self.n_keep

    def filter(self, triplets):
        t2 = np.copy(triplets)
        for i, t in enumerate(triplets):
            t2[i][1] = self.d[t[1]]

        print(t2)
        return t2

