import os
import re
from tqdm import trange
from rank_bm25 import BM25Okapi
from utils import load_json, save_json
from sklearn.metrics import ndcg_score
from collection.ncbi_api import search_term


en_punc = '.,<>?/\\[]{};:\'\"|=+-_()*&^%$#@!~`\n\t '
mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
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
    path_base, ans_base = 'manual/manual_files', 'manual/manual_new'
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


def main_bm25():
    """
    BM25 baseline for BioRADR
    head + tail: 1 60.74; 5 54.21; 10 55.28; 20 57.14; 50 64.47; -1 82.02
    heads + tails: 1 50.85; 5 49.34; 10 49.94; 20 52.07; 50 62.53; -1 79.9
    heads + tails (set): 1 44.27; 5 45.57; 10 46.75; 20 50.8; 50 61.51; -1 79.26
    """
    ignores = [0, 1, 25, 26]
    ndcg = {1: [], 5: [], 10: [], 20: [], 50: [], -1: []}
    for idx in trange(50, desc='bm25'):
        if idx in ignores:
            continue
        head, tail, corpus, labels, ret_pmids, is_pmc, heads, tails = get_query_corpus_label(idx)
        bm25 = BM25Okapi(corpus)
        doc_scores = bm25.get_scores(head + tail)
        assert len(doc_scores) == len(corpus)
        for key in ndcg.keys():
            n = key if key > 0 else None
            score = ndcg_score(y_true=[labels], y_score=[doc_scores], k=n, ignore_ties=False)
            ndcg[key].append(score)
    for key, val in ndcg.items():
        assert len(val) == 46
        print(key, round(100 * sum(val) / len(val), 2))


def main_pmc():
    """
    search engine baseline for BioRADR
    PMC: 1 50.3, 5 49.06, 10 49.97, 20 52.52, 50 61.91, -1 80.1
    """
    ignores, cache = [0, 1, 25, 26], []
    for idx in range(50):
        if idx in ignores:
            continue
        head, tail, corpus, labels, ret_pmids, is_pmc, heads, tails = get_query_corpus_label(idx)
        head_query = ' OR '.join(' '.join(head_token) for head_token in heads)
        tail_query = ' OR '.join(' '.join(tail_token) for tail_token in tails)
        whole_query = f'({head_query}) AND ({tail_query})'
        batch_size, cur_start = 100, 0
        pmid_to_score, cur_score, actual_remain = {}, len(ret_pmids), len(ret_pmids)
        while True:
            results = search_term(whole_query, db="pmc" if is_pmc else "pubmed",
                                  ret_start=cur_start, ret_max=batch_size)
            results = [int(pmid) for pmid in results]
            cur_start += batch_size
            for pmid in results:
                if pmid in ret_pmids:
                    assert pmid not in pmid_to_score
                    pmid_to_score[pmid] = cur_score
                    cur_score -= 1
            actual_remain = len([pmid for pmid in ret_pmids if pmid not in pmid_to_score])
            print(f'processed index: {idx:02}, start: {cur_start:05}, score: {cur_score:03}, '
                  f'actual remain: {actual_remain:03}', end='\r')
            if actual_remain == 0 or len(results) < batch_size:
                break
        print()
        remain_pmids = [pmid for pmid in ret_pmids if pmid not in pmid_to_score]
        print(idx, actual_remain, remain_pmids)
        cache.append({"pmid_to_score": pmid_to_score, "remain_pmids": remain_pmids})
        save_json(cache, 'CTDRED/test_cache.json')

    cache, cur_cid = load_json('CTDRED/test_cache.json'), 0
    assert len(cache) == 50 - len(ignores)
    ndcg = {1: [], 5: [], 10: [], 20: [], 50: [], -1: []}
    for idx in range(50):
        if idx in ignores:
            continue
        pmid_to_score, remain_pmids = cache[cur_cid]["pmid_to_score"], cache[cur_cid]["remain_pmids"]
        assert len(remain_pmids) == 0
        head, tail, corpus, labels, ret_pmids, is_pmc, heads, tails = get_query_corpus_label(idx)
        doc_scores = [pmid_to_score[str(pmid)] for pmid in ret_pmids]
        for key in ndcg.keys():
            n = key if key > 0 else None
            score = ndcg_score(y_true=[labels], y_score=[doc_scores], k=n, ignore_ties=False)
            ndcg[key].append(score)
        cur_cid += 1
    for key, val in ndcg.items():
        assert len(val) == 46
        print(key, round(100 * sum(val) / len(val), 2))


if __name__ == '__main__':
    # main_bm25()
    main_pmc()
