import pickle
import networkx as nx
import random
from torch.utils.data import Dataset, DataLoader
from transformers import DefaultDataCollator

import numpy as np
from tqdm import tqdm
import os

from typing import Any, Dict, List, Literal, Union, Tuple

import numpy as np
import os


class TripletDataset(Dataset):
    def __init__(self, graph, path):
        self.graph = graph
        self.triplets = list(self.generate_triplets())
        random.shuffle(self.triplets)

        if os.path.exists(path):
            self.matrix = pickle.load(open(path, 'rb'))
        else:
            self.matrix = self.generate_matrix()

        self.indices = self.indices()

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

    def generate_triplets(self):
        for node, degree in self.graph.out_degree():
            if (
                degree >= 1
                and len(node) > 1
            ):
                for child in self.graph.successors(node):
                        grandchildren = list(self.graph.successors(child))
                        if grandchildren:
                            for gch in grandchildren:
                                yield (node, child, gch)
                        else:
                            continue

    def get_paths(self):
        lengths = {}
        leng = list(nx.shortest_path_length(self.graph))
        for i in leng:
            lengths[i[0]] = i[1]

        return lengths
    
    def check_neighbour(self, lengths, triplet1, triplet2):
        pairs = [(triplet1[0], triplet2[0]), (triplet1[0], triplet2[2]), (triplet1[2], triplet2[0]), (triplet1[2], triplet2[2])]
        for pair in pairs:
            try:
                if lengths[pair[0]][pair[1]] < 2:
                    return False
            except:
                continue
        return True
    
    def generate_matrix(self):
        paths = self.get_paths()

        matrix = np.empty((len(self.triplets), len(self.triplets)), dtype='bool')
        overall_time = 0
        for i in tqdm(range(len(self.triplets))):

            for j in range(i, len(self.triplets)):
                value = self.check_neighbour(paths, self.triplets[i], self.triplets[j])

                if value:
                    matrix[i, j] = False
                    matrix[j, i] = False
                else:
                    matrix[i, j] = True
                    matrix[j ,i] = True

        pickle.dump(matrix, open('matrix_final.pickle', 'wb'))
        return matrix
    
    def indices(self):
        indices = {}
        for i in range(len(self.triplets)):
            indices[self.triplets[i]] = i

        return indices


class GraphDataCollator(DefaultDataCollator):
    def __init__(self, matrix, indices, tokenizer):
        super().__init__()
        self.matrix = matrix
        self.indices = indices
        self.tokenizer = tokenizer

    def __call__(self, features):

        suitable_triplets = []
        for triplet in features:
            flag = True
            first_index = self.indices[triplet]

            for second_tr in features:
                second_index = self.indices[second_tr]

                if triplet != second_tr:
                    result = self.matrix[first_index][second_index]
                    if result:
                        flag = False
                        break
            if flag:
            
                suitable_triplets.append(triplet)
                
        first_pair, second_pair = [], []   
        
        for triplet in suitable_triplets:
            first_pair.append(f'{triplet[0]}-{triplet[1]}')
            second_pair.append(f'{triplet[1]}-{triplet[2]}')
        
        final  = first_pair + second_pair
        tokenized_final = self.tokenizer(final, padding=True, return_tensors='pt')
    
        return tokenized_final

