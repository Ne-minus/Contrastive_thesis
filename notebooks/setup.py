from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import networkx as nx
from typing import Any, Dict, List, Literal, Union, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import  BertModel
from random import sample


class TripletDataset(Dataset):
    def __init__(self, G, tokenizer, max_length=32):
        self.graph = G
        self.triplets = self.sample_triplets()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Tokenize all triplets
        self.tokenized_triplets = self.tokenize_triplets()

    def sample_triplets(self):
        triplets = []
        for node in tqdm(self.graph.nodes()):
            preds = list(self.graph.predecessors(node))
            if preds:
        
                # print(preds)
                positive = sample(preds, 1)[0].split('.')[0]
            
                flag = True
                while flag:
                    negative = sample(list(self.graph.nodes), 1)
                    if negative not in preds:
                        negative = negative[0].split('.')[0]
                        flag = False
                        
                # node_new = node.split('.')[0]
                # triplet = (f'Concept: {node_new}', f'Concept: {positive}', f'Concept: {negative}')
                triplet = (node.split('.')[0], positive, negative)
                triplets.append(triplet)
            else:
                continue
        return triplets

    def tokenize_triplets(self):
        anchor_texts = [triplet[0] for triplet in self.triplets]
        positive_texts = [triplet[1] for triplet in self.triplets]
        negative_texts = [triplet[2] for triplet in self.triplets]

        anchor_texts = self.tokenizer(anchor_texts, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        positive_texts = self.tokenizer(positive_texts, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        negative_texts = self.tokenizer(negative_texts, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

            

        return anchor_texts, positive_texts, negative_texts

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return {
            'anchor_input_ids': self.tokenized_triplets[0]['input_ids'][idx],
            'anchor_attention_mask': self.tokenized_triplets[0]['attention_mask'][idx],
            'positive_input_ids': self.tokenized_triplets[1]['input_ids'][idx],
            'positive_attention_mask': self.tokenized_triplets[1]['attention_mask'][idx],
            'negative_input_ids': self.tokenized_triplets[2]['input_ids'][idx],
            'negative_attention_mask': self.tokenized_triplets[2]['attention_mask'][idx],
        }
    

class SentenceBERT(nn.Module):
    def __init__(self, model_name):
        super(SentenceBERT, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # Extract the last hidden state
        
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        mean_pooled_output = sum_embeddings / sum_mask
        return mean_pooled_output
    

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_similarity = F.pairwise_distance(anchor, positive, p=2)
        neg_similarity = F.pairwise_distance(anchor, negative, p=2)
        loss = torch.mean(F.relu(self.margin + pos_similarity - neg_similarity))
        return loss