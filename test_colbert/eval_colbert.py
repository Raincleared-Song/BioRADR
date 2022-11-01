import os
import re
import json
import faiss
import argparse
from tqdm import trange
from colbert.data import Queries
from colbert import Indexer, Searcher
from colbert.infra import Run, RunConfig, ColBERTConfig
from sklearn.metrics import ndcg_score


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path, encoding='utf-8')
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w', encoding='utf-8')
    json.dump(obj, file)
    file.close()


def save_tsv(obj: list, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w', encoding='utf-8')
    for line in obj:
        file.write('\t'.join(str(item) for item in line) + '\n')
    file.close()


en_punc = '.,<>?/\\[]{};:\'\"|=+-_()*&^%$#@!~`\n\t '
mesh_id_to_name = load_json('../CTDRED/mesh_id_to_name.json')
label_to_rank = {
    (1, 3): 5, (1, 2): 4, (1, 1): 3,
    (0, 1): 2, (0, 2): 1, (0, 3): 0,
    (1, 2.5): 4.5, (1, 1.5): 3.5,
    (0, 1.5): 1.5, (0, 2.5): 0.5,
}


def split_clear_text(text: str):
    text = text.replace(')', ' ').replace('(', ' ')
    tokens = text.split(' ')
    tokens = [token.lower().strip(en_punc) for token in tokens]
    tokens = [token for token in tokens if token != '']
    return tokens


def get_query_corpus_label(index: int):
    """
    5 - abs from PubMed, 45 - seg from PMC
    """
    path_base, ans_base = '../manual/manual_files', '../manual/manual_new'
    is_pmc = os.path.exists(f'{path_base}/{index}_seg.txt')
    if is_pmc:
        file_name, ans_name = f'{index}_seg.txt', f'{index}_seg_ans.txt'
    else:
        file_name, ans_name = f'{index}_abs.txt', f'{index}_abs_ans.txt'
    with open(f'{path_base}/{file_name}', encoding='utf-8') as fin:
        lines = fin.readlines()
    tokens = lines[1].split('\t')
    head, tail = split_clear_text(mesh_id_to_name[tokens[0][1:-1]]), split_clear_text(mesh_id_to_name[tokens[1][1:-1]])
    document_cnt = int(tokens[2][1:-1])
    ret_corpus, cur_document, ret_pmids = [], [], []
    head_names, tail_names = {' '.join(head)}, {' '.join(tail)}
    for line in lines:
        if line.startswith('======') and 'end' in line:
            ret_corpus.append(cur_document)
            assert len(cur_document) > 1, str(cur_document)
            cur_document = []
        if line.startswith('SENTENCE') or line.startswith('*** SENTENCE'):
            pos = line.find(':')
            assert pos != -1
            text = line[pos+2:]
            cur_document += split_clear_text(text)
            head_mentions = re.findall(f'\[\[([^]]+)]]', line)
            for mention in head_mentions:
                head_names.add(' '.join(split_clear_text(mention)))
            tail_mentions = re.findall(f'<<([^>]+)>>', line)
            for mention in tail_mentions:
                tail_names.add(' '.join(split_clear_text(mention)))
        if line.startswith('pmid:'):
            if is_pmc:
                pmd_id = line.split('_')[1]
                assert pmd_id.startswith('PMC'), line
                ret_pmids.append(int(pmd_id[3:]))
            else:
                pmid = line.split(' ')[1]
                ret_pmids.append(int(pmid))
    assert len(ret_corpus) == len(ret_pmids) == document_cnt, \
        f'{index}: {len(ret_corpus)} | {len(ret_pmids)} | {document_cnt}'

    with open(f'{ans_base}/{ans_name}', encoding='utf-8') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    assert len(lines) == document_cnt
    labels = []
    for line in lines:
        tokens = line.split(',')
        assert len(tokens) == 3, str(tokens) + ' | ' + str(index)
        h, r = int(tokens[1]), float(tokens[2])
        labels.append(label_to_rank[(h, r)])
    head_names.remove(' '.join(head))
    tail_names.remove(' '.join(tail))
    head_names = [head] + list(mention.split(' ') for mention in head_names)
    tail_names = [tail] + list(mention.split(' ') for mention in tail_names)
    return head, tail, ret_corpus, labels, ret_pmids, is_pmc, head_names, tail_names


def indexing(group_id, corpus, head, tail):
    fout = open(f'bioradr/tsv/{group_id}_collection.tsv', 'w', encoding='utf-8')
    for did, document in enumerate(corpus):
        sentence = ' '.join(document)
        fout.write(str(did) + '\t' + sentence + '\n')
    fout.close()
    fout = open(f'bioradr/tsv/{group_id}_query.tsv', 'w', encoding='utf-8')
    fout.write('0\t' + ' '.join(head + tail) + '\n')
    fout.close()

    with Run().context(RunConfig(
            nranks=1,
            experiment="BioRADR",
            gpus=1,
            amp=False,
            overwrite=True,
    )):
        config = ColBERTConfig(
            nbits=2,
            bsize=8,
            query_maxlen=64,
            doc_maxlen=512,
            reranker=True,
        )
        indexer = Indexer(checkpoint="colbertv2.0", config=config)
        indexer.config.save(f'bioradr/configs/{group_id}_config.json', overwrite=True)
        indexer.index(name=f"BioRADR-group-{group_id}",
                      collection=f'bioradr/tsv/{group_id}_collection.tsv', overwrite=True)


def querying(group_id):
    with Run().context(RunConfig(
            nranks=1,
            experiment="BioRADR",
            gpus=1,
            amp=False,
            overwrite=True,
    )):
        config = ColBERTConfig(
            nbits=2,
            bsize=8,
            query_maxlen=64,
            doc_maxlen=512,
            reranker=True,
            is_rerank=True,
        )
        searcher = Searcher(index=f"BioRADR-group-{group_id}", config=config, checkpoint="colbertv2.0")
        queries = Queries(f'bioradr/tsv/{group_id}_query.tsv')
        ranking = searcher.search_all(queries, k=1000)
        print(f'returned document count: {len(ranking.flat_ranking)}')
        return ranking.flat_ranking


def main():
    """
    1 61.23; 5 56.06; 10 56.79; 20 57.06; 50 65.20; -1 81.78 (k=100)
    1 61.23; 5 56.01; 10 56.79; 20 57.33; 50 65.42; -1 81.90 (k>=1000)
    1 61.23; 5 56.26; 10 56.80; 20 57.34; 50 65.37; -1 81.98 (k>=1000, re-rank)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='a single GPU rank', type=int, default=0)
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    ignores = [0, 1, 25, 26]  # left out for validation
    ndcg = {1: [], 5: [], 10: [], 20: [], 50: [], -1: []}
    for group_id in trange(50, desc='groups'):
        if group_id in ignores:
            continue
        head, tail, ret_corpus, labels, ret_pmids, is_pmc, head_names, tail_names = get_query_corpus_label(group_id)
        indexing(group_id, ret_corpus, head, tail)
        ranking_res = querying(group_id)
        save_tsv(ranking_res, f'bioradr/results/{group_id}_result.tsv')
        cur_score, doc_scores = 100, [-100 for _ in range(len(ret_corpus))]
        assert len(ranking_res) == len(ret_corpus)
        for qid, doc_id, order, score in ranking_res:
            assert qid == 0 and cur_score + order == 101
            doc_scores[doc_id] = cur_score
            cur_score -= 1
        assert all(score > -100 for score in doc_scores)
        for n in ndcg.keys():
            score = ndcg_score(y_true=[labels], y_score=[doc_scores], k=n, ignore_ties=False)
            ndcg[n].append(score)
    save_json(ndcg, 'bioradr/result_ndcg.json')
    for key, val in ndcg.items():
        n_key = str(key) if key != -1 else 'All'
        assert len(val) == 46
        print(f'NDCG@{n_key}', round(100 * sum(val) / len(val), 2))


if __name__ == '__main__':
    main()
