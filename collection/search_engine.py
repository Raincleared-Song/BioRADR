import sys
import copy
import json
import torch
import spacy
import traceback
import numpy as np
from torch.autograd import Variable
from ncbi_api import spell_term, is_mesh_id
from search_initialize import init_arg_model, init_data
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker

import warnings
warnings.filterwarnings(action='ignore', module='.*?spacy.*?')


GLOBAL = {
    'part_names': ['train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test',
                   'negative_train_extra', 'negative_dev_extra', 'negative_test_extra', 'ctd_extra'],
    'linker': spacy.language.Language(),
    'pair_to_docs': {},
    'chemical_to_tail_doc': {},
    'disease_to_head_doc': {},
    'mesh_id_to_name': {},
    'pmid_to_part': {},
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


def get_documents_by_pmids(pmids: list, require_all: bool):
    """
    :param pmids: list of int pmids
    :param require_all if every pmid should be loaded
    :return: DocRED format documents
    """
    global GLOBAL
    pmid_to_part = GLOBAL['pmid_to_part']
    if require_all:
        for pmid in pmids:
            assert pmid in pmid_to_part
    else:
        pmids = [pmid for pmid in pmids if pmid in pmid_to_part]
    ret = [{} for _ in range(len(pmids))]
    part_names = GLOBAL['part_names']
    pmid2idx = {pmid: i for i, pmid in enumerate(pmids)}
    part2pmids = {pid: [] for pid in range(len(part_names))}
    for pmid in pmids:
        part2pmids[pmid_to_part[pmid]].append(pmid)
    for pid, val in part2pmids.items():
        if len(val) <= 0:
            continue
        data = load_json(f'CTDRED/{part_names[pid]}_binary_pos.json')
        data = {doc['pmid']: doc for doc in data}
        for pmid in val:
            ret[pmid2idx[pmid]] = data[pmid]
        del data
    return ret


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


def process(entity1: str, entity2: str):
    global GLOBAL
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
        if pair not in pair_to_docs:
            return []
        docs = pair_to_docs[pair][0] + pair_to_docs[pair][1]
        docs = get_documents_by_pmids(docs, require_all=True)
        ret = [(model_score(docs, cid1, cid2), cid1, get_name_by_mesh_id(cid1), cid2, get_name_by_mesh_id(cid2))]
        return ret
    # entity1 == '' or entity2 == ''
    if entity1 == '':
        # find heads of entity2
        assert is_mesh_id(cid2)
        disease_to_head_doc = GLOBAL['disease_to_head_doc']
        if cid2 not in disease_to_head_doc:
            return []
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
            return []
        candidates = []
        for cid2, _ in chemical_to_tail_doc[cid1]:
            # 只取 MeSH 的实体
            if is_mesh_id(cid2):
                candidates.append(cid2)
    candidate_num = len(candidates)
    if candidate_num == 0:
        return 'no candidate head entities'
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
        return [([], cid1, cname1, cid2, cname2)]
    docs = pair_to_docs[candidate_pair][0] + pair_to_docs[candidate_pair][1]
    docs = get_documents_by_pmids(docs, require_all=False)
    return [(model_score(docs, cid1, cid2), cid1, cname1, cid2, cname2)]


def repeat_input(info: str, restrict=None, int_range=None):
    cont = ''
    while cont == '':
        cont = input(info).strip()
        if restrict is not None and len(restrict) > 0 and cont not in restrict:
            print(f'input should be in {restrict}')
            cont = ''
        if int_range is not None:
            assert len(int_range) == 2
            if not cont.isdigit() or int(cont) >= int_range[1] or int(cont) < int_range[0]:
                print(f'input should be an integer and in {int_range}')
    return cont


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def save_json(obj: object, path: str):
    print(f'saving file {path} ......')
    file = open(path, 'w')
    json.dump(obj, file)
    file.close()


def init_all():
    global GLOBAL
    args, model = init_arg_model()
    GLOBAL['args'] = args
    GLOBAL['model'] = model

    # init scispacy
    assert torch.cuda.is_available()
    spacy.require_gpu(0)
    nlp = spacy.load('en_core_sci_lg')
    nlp.add_pipe('abbreviation_detector')
    nlp.add_pipe('scispacy_linker', config={'resolve_abbreviations': True, 'linker_name': 'mesh'})
    GLOBAL['linker'] = nlp

    # init pair to docs (in data), entity_to_other (in ctd)
    GLOBAL['pair_to_docs'] = load_json('CTDRED/pair_to_docs.json')
    GLOBAL['chemical_to_tail_doc'] = load_json('CTDRED/chemical_to_tail_doc.json')
    GLOBAL['disease_to_head_doc'] = load_json('CTDRED/disease_to_head_doc.json')
    GLOBAL['mesh_id_to_name'] = load_json('CTDRED/mesh_id_to_name.json')
    tmp = {}
    for key, val in load_json('CTDRED/pmid_to_part.json').items():
        tmp[int(key)] = val
    GLOBAL['pmid_to_part'] = tmp
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
                for doc, score in result:
                    print(doc['pmid'], score, sep='\t')
        except Exception as err:
            print(f'Something Error: [{entity1}] [{entity2}]', file=sys.stderr)
            traceback.print_exception(type(err), err, sys.exc_info()[2])


if __name__ == '__main__':
    init_all()
    main_loop()
