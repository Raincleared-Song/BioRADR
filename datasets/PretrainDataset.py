from torch.utils.data import Dataset
from config import ConfigPretrain as Config
from utils import load_json, sigmoid
import numpy as np
import random


class PretrainDataset(Dataset):
    def __init__(self, task: str, mode: str):
        assert task == 'pretrain' and mode != 'test'

        self.data, self.reserved_did, self.pair_to_triple = load_pretrain_data(mode)
        self.entity_pairs = list(self.pair_to_triple.keys())

        if mode != 'train':
            self.total_len = 200 * Config.batch_size[mode] * Config.train_steps
        else:
            self.total_len = 1000 * Config.batch_size[mode] * Config.train_steps

    def __getitem__(self, item):
        if random.random() < Config.same_pair_ratio:
            pair = random.choice(self.entity_pairs)  # same entity
            query, doc = random.sample(self.pair_to_triple[pair], 2)
            return {
                'doc1': self.data[query[0]],  # whole data
                'doc2': self.data[doc[0]],
                'pair1': (query[1], query[2]),  # entity id pair
                'pair2': (doc[1], doc[2])
            }

        else:
            query, doc = random.sample(self.reserved_did, 2)  # sample 2 documents
            query, doc = self.data[query], self.data[doc]
            while len(query['vertexSet']) < 5 or len(doc['vertexSet']) < 5 or \
                    len(query['labels']) == 0 or len(doc['labels']) == 0:
                query, doc = random.sample(self.reserved_did, 2)
                query, doc = self.data[query], self.data[doc]
            return {
                'doc1': query,
                'doc2': doc,
                'pair1': (0, 0),
                'pair2': (0, 0)
            }

    def __len__(self):
        return self.total_len


def load_pretrain_data(mode: str):
    data = load_json(Config.data_path[mode])
    positive_pairs = set()

    if Config.use_score:
        scores = sigmoid(np.load(Config.score_path[mode]))
        titles = load_json(Config.title_path[mode])
        if isinstance(titles, list):
            titles = {title: i for i, title in enumerate(titles)}
        for did, doc in enumerate(data):
            entities = doc['vertexSet']
            entity_num = len(entities)
            # abandon those documents with less than 5 entities
            if entity_num < 5:
                continue
            t_idx = titles[int(doc['pmid'])]
            if isinstance(t_idx, int):
                score = scores[t_idx]
            else:
                score = scores[t_idx[0]:t_idx[1]]
            pair_to_score = []
            for i in range(entity_num):
                for j in range(entity_num):
                    if i == j:
                        continue
                    pair_to_score.append(((i, j), score[len(pair_to_score)]))
            pair_to_score.sort(key=lambda x: x[1], reverse=True)

            if Config.score_threshold is None:
                # reserve highest 30 pairs
                reserved_pairs = set([item[0] for item in pair_to_score[:Config.kept_pair_num]])
            else:
                # reserve pairs with high score
                reserved_pairs = set([item[0] for item in pair_to_score if item[1] > Config.score_threshold])

            for lid, lab in enumerate(doc['labels']):
                if (lab['h'], lab['t']) in reserved_pairs:
                    positive_pairs.add((did, lab['h'], lab['t']))
                    data[did]['labels'][lid]['exist'] = True
                else:
                    data[did]['labels'][lid]['exist'] = False
    else:
        for did, doc in enumerate(data):
            entities = doc['vertexSet']
            entity_num = len(entities)
            # abandon those documents with less than 5 entities
            if entity_num < 5:
                continue
            for lid, lab in enumerate(doc['labels']):
                positive_pairs.add((did, lab['h'], lab['t']))
                data[did]['labels'][lid]['exist'] = True

    pair_dict = load_json(Config.pair2triple_path)  # Dict: 'ent&ent -> List[(did, h, t)]'

    reserved_did = set(random.sample(range(len(data)),
                                     int(Config.data_ratio * len(data))))

    pair_to_triple = {}
    for pair, triples in pair_dict.items():
        cur = []
        for triple in triples:
            triple = triple[:3]
            if tuple(triple) in positive_pairs and triple[0] in reserved_did:
                cur.append(triple)
        if len(cur) > 1:
            pair_to_triple[pair] = cur

    return data, list(reserved_did), pair_to_triple
