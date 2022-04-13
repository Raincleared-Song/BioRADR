import sys
import copy
import torch
import spacy
import traceback
import numpy as np
from torch.autograd import Variable
from .ncbi_api import spell_term, is_mesh_id, search_get_pubmed, pubtator_to_docred
from .search_utils import load_json, repeat_input
from .search_initialize import init_args, init_model, init_data
from .search_db import init_db, get_documents_by_pmids
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

import warnings
warnings.filterwarnings(action='ignore', module='.*?spacy.*?')


GLOBAL = {
    'api_threshold': 50,  # 少于这个阈值则使用搜索 api 兜底
    'na_score': -1000,    # 不包含实体对时的默认得分
    'linker': spacy.language.Language(),
    'pair_to_docs': {},
    'chemical_to_tail_doc': {},
    'disease_to_head_doc': {},
    'mesh_id_to_name': {},
    'model': None,
    'args': None
}


def spell_and_link(entity: str):
    # TODO linking 准确率不够，可能要考虑所有情况 methylprednisolone acetate & pain
    global GLOBAL
    entity = spell_term(entity)
    nlp = GLOBAL['linker']
    doc = nlp(entity)
    if len(doc.ents) != 1:
        return '', f'{len(doc.ents)} entities found'
    kbs = doc.ents[0]._.kb_ents
    print(entity, doc.ents[0]._.kb_ents, sep='\t')
    if len(kbs) == 0:
        return '', 'link failed'
    cid = kbs[0][0]
    return entity, cid


def gen_score(args, model, dataset):
    model.eval()
    use_gpu = args.device.strip().lower().startswith('cuda')

    scores, titles = [], []
    for step, data in enumerate(dataset):
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                data[key] = Variable(value.to(args.device)) if use_gpu else Variable(value)

        result = model(data, 'test')
        scores += result['score'].cpu().tolist()
        titles += result['titles']

    return scores, titles


def model_score(docs: list, cid1: str, cid2: str):
    """
    score the documents containing the entity pair
    :param docs: list of documents (DocRED format)
    :param cid1: MeSH ID of head
    :param cid2: MeSH ID of tail
    :return: score_doc_list
    """
    global GLOBAL
    if len(docs) == 0:
        return []
    model, args = GLOBAL['model'], GLOBAL['args']
    assert model is not None and args is not None
    print(f'============ scoring {len(docs)} documents ============')
    docs = [copy.deepcopy(doc) for doc in docs]
    # process documents
    for doc in docs:
        new_vertex_set, new_cids = [], []
        del doc['labels']
        assert len(doc['vertexSet']) == len(doc['cids'])
        for entity, cid in zip(doc['vertexSet'], doc['cids']):
            if cid in (cid1, cid2):
                new_vertex_set.append(entity)
                new_cids.append(cid)
        doc['vertexSet'] = new_vertex_set
        doc['cids'] = new_cids
        try:
            assert len(doc['vertexSet']) == len(doc['cids']) == 2
        except AssertionError as err:
            from IPython import embed
            embed()
            raise err
        assert new_vertex_set[0][0]['type'] != new_vertex_set[1][0]['type']

    data_loader = init_data(args, docs)
    with torch.no_grad():
        scores, titles = gen_score(args, model, data_loader)

    scores = np.vstack(scores)
    assert scores.shape[0] == len(titles) == len(docs) and scores.shape[1] == 1
    ret = []
    for pmid, doc in zip(titles, docs):
        assert pmid == doc['pmid']
    for idx, doc in enumerate(docs):
        ret.append((doc, scores[idx, 0]))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def get_name_by_mesh_id(mesh_id: str):
    global GLOBAL
    mesh_id_to_name = GLOBAL['mesh_id_to_name']
    if mesh_id not in mesh_id_to_name:
        return 'UNK'
    else:
        return mesh_id_to_name[mesh_id]


def get_supplementary_documents(cid1, cid2, cname1, cname2, filter_pmids: set):
    """从 PubMed API 取得补充语料"""
    global GLOBAL
    api_threshold = GLOBAL['api_threshold']
    if len(filter_pmids) >= api_threshold:
        return [], []
    docs, docs_null = search_get_pubmed([cname1, cname2], pmid_filter=filter_pmids,
                                        ent_pair=(cid1, cid2), ret_max=api_threshold)
    exist_set = set()
    ret_pos, ret_neg = [], []
    for did, val in docs.items():
        try:
            did = int(did)
            assert did not in filter_pmids and did not in exist_set
            ret_pos.append(pubtator_to_docred(val, []))
            exist_set.add(did)
        except Exception as err:
            print('pubtator_to_docred error:', did, file=sys.stderr)
            traceback.print_exception(type(err), err, sys.exc_info()[2], file=sys.stderr)
    for val in docs_null:
        did = int(val['pmid'])
        try:
            assert did not in filter_pmids and did not in exist_set
            ret_neg.append(pubtator_to_docred(val, []))
            exist_set.add(did)
        except Exception as err:
            print('pubtator_to_docred error:', did, file=sys.stderr)
            traceback.print_exception(type(err), err, sys.exc_info()[2], file=sys.stderr)
    return ret_pos, ret_neg


def process(entity1: str, entity2: str):
    global GLOBAL
    api_threshold, na_score = GLOBAL['api_threshold'], GLOBAL['na_score']
    cid1 = cid2 = ''
    if entity1 != '':
        entity1, cid1 = spell_and_link(entity1)
        if entity1 == '':
            return 'entity1 link failure'
    if entity2 != '':
        entity2, cid2 = spell_and_link(entity2)
        if entity2 == '':
            return 'entity2 link failure'
    print(f'entity1: [{cid1}] entity2: [{cid2}]')
    assert not (entity1 == '' and entity2 == '')
    if entity1 != '' and entity2 != '':
        assert is_mesh_id(cid1) and is_mesh_id(cid2)
        pair = cid1 + '&' + cid2
        pair_to_docs = GLOBAL['pair_to_docs']
        cname1, cname2 = get_name_by_mesh_id(cid1), get_name_by_mesh_id(cid2)
        if pair not in pair_to_docs:
            # 无对应记录，全部兜底
            sup_pos, sup_neg = get_supplementary_documents(cid1, cid2, cname1, cname2, set())
            docs = sup_pos
        else:
            docs = pair_to_docs[pair][0] + pair_to_docs[pair][1]
            docs = get_documents_by_pmids(docs, require_all=True)
            sup_pos, sup_neg = [], []
            if len(docs) < api_threshold:
                # 使用 api 进行补充
                filter_set = set([int(doc['pmid']) for doc in docs])
                sup_pos, sup_neg = get_supplementary_documents(cid1, cid2, cname1, cname2, filter_set)
            docs = docs + sup_pos
        rank_ret = model_score(docs, cid1, cid2)
        rank_ret += [(doc, na_score) for doc in sup_neg]
        return [(rank_ret, cid1, cname1, cid2, cname2)]
    # entity1 == '' or entity2 == ''
    if entity1 == '':
        # find heads of entity2
        assert is_mesh_id(cid2)
        disease_to_head_doc = GLOBAL['disease_to_head_doc']
        if cid2 not in disease_to_head_doc:
            return 'no candidate related head entities'
        candidates = []
        for cid1, _ in disease_to_head_doc[cid2]:
            # 只取 MeSH 的实体
            if is_mesh_id(cid1):
                candidates.append(cid1)
    else:
        # find tails of entity1
        assert is_mesh_id(cid1)
        chemical_to_tail_doc = GLOBAL['chemical_to_tail_doc']
        if cid1 not in chemical_to_tail_doc:
            return 'no candidate related tail entities'
        candidates = set()
        for cid2, _ in chemical_to_tail_doc[cid1]:
            # 只取 MeSH 的实体
            if is_mesh_id(cid2):
                candidates.add(cid2)
    candidates = list(candidates)
    candidate_num = len(candidates)
    if candidate_num == 0:
        return 'no candidate related head entities'
    start, end = 0, 4
    while start < candidate_num:
        to_print = []
        for idx, cid in enumerate(candidates[start:end]):
            cname = get_name_by_mesh_id(cid)
            to_print.append(f'{(start + idx):2}-{cid}: {cname}')
        print(*to_print, sep='\t')
        start, end = end, end + 4
    candidate_id = int(repeat_input('please input a candidate idx > ', int_range=(0, candidate_num)))
    if entity1 == '':
        candidate_pair = candidates[candidate_id] + '&' + cid2
    else:
        candidate_pair = cid1 + '&' + candidates[candidate_id]
    pair_to_docs = GLOBAL['pair_to_docs']
    cid1, cid2 = candidate_pair.split('&')
    cname1, cname2 = get_name_by_mesh_id(cid1), get_name_by_mesh_id(cid2)
    if candidate_pair not in pair_to_docs:
        sup_pos, sup_neg = get_supplementary_documents(cid1, cid2, cname1, cname2, set())
        docs = sup_pos
    else:
        docs = pair_to_docs[candidate_pair][0] + pair_to_docs[candidate_pair][1]
        docs = get_documents_by_pmids(docs, require_all=False)
        docs = [doc for doc in docs if doc != {}]
        sup_pos, sup_neg = [], []
        if len(docs) < api_threshold:
            # 使用 api 进行补充
            filter_set = set([int(doc['pmid']) for doc in docs])
            sup_pos, sup_neg = get_supplementary_documents(cid1, cid2, cname1, cname2, filter_set)
        docs = docs + sup_pos
    rank_ret = model_score(docs, cid1, cid2)
    rank_ret += [(doc, na_score) for doc in sup_neg]
    return [(rank_ret, cid1, cname1, cid2, cname2)]


def init_all():
    global GLOBAL
    args = init_args()

    # init scispacy
    if args.device.startswith('cuda:'):
        assert torch.cuda.is_available()
        spacy.require_gpu(int(args.device[5:]))
    nlp_init_err = None
    for _ in range(3):
        try:
            nlp = spacy.load('en_core_sci_lg')
            nlp.add_pipe('abbreviation_detector')
            nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'mesh'})
            GLOBAL['linker'] = nlp
            nlp_init_err = None
            break
        except Exception as err:
            nlp_init_err = err
            continue
    if nlp_init_err is not None:
        traceback.print_exception(type(nlp_init_err), nlp_init_err, sys.exc_info()[2])
        raise RuntimeError('scispacy initialize error')

    model = init_model(args)
    GLOBAL['args'] = args
    GLOBAL['model'] = model
    init_db()

    # init pair to docs (in data), entity_to_other (in ctd)
    GLOBAL['pair_to_docs'] = load_json('CTDRED/pair_to_docs.json')
    GLOBAL['chemical_to_tail_doc'] = load_json('CTDRED/chemical_to_tail_doc.json')
    GLOBAL['disease_to_head_doc'] = load_json('CTDRED/disease_to_head_doc.json')
    GLOBAL['mesh_id_to_name'] = load_json('CTDRED/mesh_id_to_name.json')
    print('loaded number of pairs:', len(GLOBAL['pair_to_docs']))


def main_loop():
    print('============ server initialized ============')
    while True:
        cnt = repeat_input('number of entities [1/2] or cmd [EXIT/NEXT] > ', ['1', '2', 'EXIT', 'NEXT'])
        if cnt == 'EXIT':
            break
        elif cnt == 'NEXT':
            continue
        if cnt == '1':
            entity1 = repeat_input(f'input entity type [c/d] or cmd > ', ['c', 'd'])
        else:
            entity1 = repeat_input('input chemical or cmd > ')
        if entity1 == 'EXIT':
            break
        elif entity1 == 'NEXT':
            continue
        if cnt == '1':
            entity2 = repeat_input(f'input entity or cmd > ')
        else:
            entity2 = repeat_input('input disease or cmd > ')
        if entity2 == 'EXIT':
            break
        elif entity2 == 'NEXT':
            continue
        try:
            if cnt == '1':
                if entity1 == 'c':
                    ret = process(entity2, '')
                else:
                    ret = process('', entity2)
            else:
                ret = process(entity1, entity2)
            if isinstance(ret, str):
                print('============', ret, '============')
                continue
            for result, cid1, cname1, cid2, cname2 in ret:
                print('============', cid1, cname1, '||', cid2, cname2, '============')
                for idx, (doc, score) in enumerate(result):
                    print(f'{idx:03}.')
                    print(doc['pmid'], score, sep='\t')
                    print('---', doc['title'].strip())
                    # print sentences
                    cid2idx = {cid: i for i, cid in enumerate(doc['cids'])}
                    mentions = []
                    # 可能不包含全部两个实体 (API 补充)
                    if cid1 in cid2idx:
                        mentions += [(mention, 0) for mention in doc['vertexSet'][cid2idx[cid1]]]
                    if cid2 in cid2idx:
                        mentions += [(mention, 1) for mention in doc['vertexSet'][cid2idx[cid2]]]
                    star_sents = {}
                    for mention, eid in mentions:
                        sid = mention['sent_id']
                        if sid not in star_sents:
                            star_sents[sid] = []
                        star_sents[sid].append((mention['pos'], eid))
                    for sid, sent in enumerate(doc['sents']):
                        starts, ends = [{}, {}], [{}, {}]
                        if sid in star_sents:
                            print('***', end=' ')
                            for pos, eid in star_sents[sid]:
                                starts[eid].setdefault(pos[0], 0)
                                starts[eid][pos[0]] = starts[eid][pos[0]] + 1
                                ends[eid].setdefault(pos[1], 0)
                                ends[eid][pos[1]] = ends[eid][pos[1]] + 1
                        tokens = []
                        sent.append('')
                        for tid, token in enumerate(sent):
                            # head entity: [[/]], tail entity <</>>
                            if tid in ends[0]:
                                tokens += [']]' for _ in range(ends[0][tid])]
                            if tid in ends[1]:
                                tokens += ['>>' for _ in range(ends[1][tid])]
                            if tid in starts[0]:
                                tokens += ['[[' for _ in range(starts[0][tid])]
                            if tid in starts[1]:
                                tokens += ['<<' for _ in range(starts[1][tid])]
                            token = token.strip()
                            if token != '':
                                tokens.append(token)
                        text = ' '.join(tokens)
                        print(f'sent {sid}: {text}')
                    print()
        except Exception as err:
            print(f'Something Error: [{entity1}] [{entity2}]', file=sys.stderr)
            traceback.print_exception(type(err), err, sys.exc_info()[2])


if __name__ == '__main__':
    init_all()
    main_loop()
