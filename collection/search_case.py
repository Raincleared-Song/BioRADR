import os
import sys
import copy
import torch
import spacy
import logging
import traceback
import numpy as np
from deprecated import deprecated
from torch.autograd import Variable
from xml.sax import SAXParseException
from .search_utils import load_json, repeat_input, setup_logger, is_mesh_id
from .ncbi_api import search_get_pubmed, pubtator_to_docred, spell_term, summary_uids, search_term
from .search_initialize import init_args, init_model, init_data
from scispacy.linking import EntityLinker
# from scispacy.abbreviation import AbbreviationDetector


GLOBAL = {
    'linker': spacy.language.Language(),
    'mesh_id_to_name': {},
    'chemical_to_tail_doc': {},
    'disease_to_head_doc': {},
    'model': None,
    'args': None,
    'logger': logging.getLogger('server'),
    'na_score': -1000,    # 不包含实体对时的默认得分
    'search_count': 100,  # 从 PubMed 搜索返回的最大文档数
}


def init_spacy_gpu(args):
    if args.device.startswith('cuda:'):
        assert torch.cuda.is_available()
        spacy.require_gpu(int(args.device[5:]))


def init_all():
    global GLOBAL
    args = init_args()
    setup_logger('server', args.log_file, 'a', '[%(asctime)s]: %(levelname)s: %(message)s', logging.INFO)

    # init scispacy
    init_spacy_gpu(args)
    nlp_init_err = None
    for _ in range(3):
        try:
            nlp = spacy.load('en_core_sci_lg')
            # nlp.add_pipe('abbreviation_detector')
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
    GLOBAL['mesh_id_to_name'] = load_json('CTDRED/mesh_id_to_name.json')
    GLOBAL['chemical_to_tail_doc'] = load_json('CTDRED/chemical_to_tail_doc.json')
    GLOBAL['disease_to_head_doc'] = load_json('CTDRED/disease_to_head_doc.json')


@deprecated('this function is depreciated and does not work well')
def link_entity(entity: str, spell=True):
    global GLOBAL
    logger = GLOBAL['logger']
    if spell:
        try:
            entity = spell_term(entity)
        except SAXParseException:
            pass
    nlp = GLOBAL['linker']
    doc = nlp(entity)
    if len(doc.ents) != 1:
        return '', f'{len(doc.ents)} entities found'
    kbs = doc.ents[0]._.kb_ents
    logger.info(f'{entity}: {doc.ents[0]._.kb_ents}')
    if len(kbs) == 0:
        return '', 'link failed'
    cid = kbs[0][0]
    return entity, cid


def link_entity_by_api(entity: str, spell=True):
    if spell:
        try:
            entity = spell_term(entity)
        except SAXParseException:
            pass
    uids = search_term(entity)
    if len(uids) == 0:
        return '', '0 entities found'
    target_uid = uids[0]
    if len(target_uid) < 2:
        return '', 'invalid uid returned'
    cid = chr(int(target_uid[:2])) + target_uid[2:]
    return entity, cid


def get_name_by_mesh_id(mesh_id: str):
    global GLOBAL
    mesh_id_to_name = GLOBAL['mesh_id_to_name']
    if mesh_id not in mesh_id_to_name:
        return 'UNK'
    else:
        return mesh_id_to_name[mesh_id]


def search_pubmed_documents(cid1, cid2, cname1, cname2, filter_pmids: set):
    """从 PubMed API 取得语料"""
    global GLOBAL
    logger = GLOBAL['logger']
    docs, docs_null = search_get_pubmed([cname1, cname2], pmid_filter=filter_pmids,
                                        ent_pair=(cid1, cid2), ret_max=GLOBAL['search_count'])
    exist_set = set()
    ret_pos, ret_neg = [], []
    for did, val in docs.items():
        try:
            did = int(did)
            assert did not in filter_pmids and did not in exist_set
            ret_pos.append(pubtator_to_docred(val, []))
            exist_set.add(did)
        except Exception as err:
            logger.error(f'pubtator_to_docred error: {did}')
            logger.error(''.join(traceback.format_exception(type(err), err, sys.exc_info()[2])))
    for val in docs_null:
        did = int(val['pmid'])
        try:
            assert did not in filter_pmids and did not in exist_set
            ret_neg.append(pubtator_to_docred(val, []))
            exist_set.add(did)
        except Exception as err:
            logger.error(f'pubtator_to_docred error: {did}')
            logger.error(''.join(traceback.format_exception(type(err), err, sys.exc_info()[2])))
    return ret_pos, ret_neg


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
    logger = GLOBAL['logger']
    if len(docs) == 0:
        return []
    model, args = GLOBAL['model'], GLOBAL['args']
    assert model is not None and args is not None
    logger.info(f'scoring {len(docs)} documents')
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
        assert str(pmid) == str(doc['pmid'])
    for idx, doc in enumerate(docs):
        ret.append((doc, scores[idx, 0]))
    ret.sort(key=lambda x: x[1], reverse=True)
    return ret


def process_one_entity(key: str, offset=0, count=20):
    global GLOBAL
    logger = GLOBAL['logger']
    init_spacy_gpu(GLOBAL['args'])
    if key == '':
        return 'empty key'
    key, key_cid = link_entity_by_api(key)
    if key == '':
        return 'key link failure'
    assert is_mesh_id(key_cid)
    chemical_to_tail_doc = GLOBAL['chemical_to_tail_doc']
    disease_to_head_doc = GLOBAL['disease_to_head_doc']
    candidate_cnt = {}
    if key_cid in chemical_to_tail_doc:
        for cid2, _ in chemical_to_tail_doc[key_cid]:
            # 只取 MeSH 的实体
            if is_mesh_id(cid2):
                candidate_cnt.setdefault(cid2, 0)
                candidate_cnt[cid2] += 1
        is_head = True
    elif key_cid in disease_to_head_doc:
        for cid1, _ in disease_to_head_doc[key_cid]:
            # 只取 MeSH 的实体
            if is_mesh_id(cid1):
                candidate_cnt.setdefault(cid1, 0)
                candidate_cnt[cid1] += 1
        is_head = False
    else:
        return 'no entity related in MeSH'
    # sort by count
    candidate_cnt = sorted(list(candidate_cnt.items()), key=lambda x: x[1], reverse=True)
    candidate_cids = [cid for cid, _ in candidate_cnt]
    candidates = [(str(ord(cid[0])) + cid[1:]) for cid in candidate_cids]
    if offset >= len(candidates):
        return 'page index exceeded'
    page_num = (len(candidates) + count - 1) // count
    results = []
    batch_size = 20
    start, end = 0, batch_size
    candidates = candidates[offset:(offset+count)]
    candidate_cids = candidate_cids[offset:(offset+count)]
    while start < len(candidates):
        cur_candidates = candidates[start:end]
        cur_candidate_cids = candidate_cids[start:end]
        try:
            data = summary_uids(cur_candidates)
        except KeyError:
            data = {}
        for uid, cid in zip(cur_candidates, cur_candidate_cids):
            if uid in data:
                scopenote, meshterms, idxlinks, _ = data[uid]
                results.append({
                    'uid': uid,
                    'mesh_id': cid,
                    'scopenote': scopenote,
                    'meshterms': meshterms,
                    'idxlinks': idxlinks,
                })
            else:
                results.append({
                    'uid': uid,
                    'mesh_id': cid,
                    'scopenote': '',
                    'meshterms': [get_name_by_mesh_id(cid)],
                    'idxlinks': [],
                })
        logger.info(f'got: {len(results):04}/{len(candidates):04}')
        start, end = end, end + batch_size
    return {
        'is_head': is_head,
        'linked_cid': key_cid,
        'results': results,
        'total_pages': page_num,
    }


def process_two_entities(entity1: str, entity2: str):
    global GLOBAL
    init_spacy_gpu(GLOBAL['args'])
    logger = GLOBAL['logger']
    na_score = GLOBAL['na_score']
    assert entity1 != '' and entity2 != ''
    entity1, cid1 = link_entity_by_api(entity1)
    if entity1 == '':
        return 'head entity link failure'
    entity2, cid2 = link_entity_by_api(entity2)
    if entity2 == '':
        return 'tail entity link failure'
    logger.info(f'entity1: [{cid1}] entity2: [{cid2}]')

    assert is_mesh_id(cid1) and is_mesh_id(cid2)
    cname1, cname2 = get_name_by_mesh_id(cid1), get_name_by_mesh_id(cid2)

    docs, sup_neg = search_pubmed_documents(cid1, cid2, cname1, cname2, set())

    rank_ret = model_score(docs, cid1, cid2)
    rank_ret += [(doc, na_score) for doc in sup_neg]
    return [(rank_ret, cid1, cname1, cid2, cname2)]


def main():
    init_all()
    os.makedirs('cases', exist_ok=True)
    print('============ server initialized ============')
    cur_success_loop_cnt = 0

    while True:
        print(f'loop {cur_success_loop_cnt} ......')
        h_name = repeat_input('input CMD or head entity > ').strip()
        if h_name == 'continue':
            continue
        elif h_name == 'exit':
            break
        elif h_name == 'clear':
            os.system('clear')
            continue
        elif h_name == 'nvidia-smi':
            os.system('nvidia-smi')
            continue
        tokens = h_name.split(' ')
        if len(tokens) == 2 and tokens[0] == 'chid' and tokens[1].isdigit():
            cur_success_loop_cnt = int(tokens[1])
            continue
        t_name = repeat_input('input CMD or tail entity > ').strip()
        if t_name == 'continue':
            continue
        elif t_name == 'exit':
            break
        elif t_name == 'clear':
            os.system('clear')
            continue
        elif t_name == 'nvidia-smi':
            os.system('nvidia-smi')
            continue
        tokens = t_name.split(' ')
        if len(tokens) == 2 and tokens[0] == 'chid' and tokens[1].isdigit():
            cur_success_loop_cnt = int(tokens[1])
            continue
        try:
            ret = process_two_entities(h_name, t_name)
        except Exception as err:
            print(f'Something Error: [{h_name}] [{t_name}]', file=sys.stderr)
            traceback.print_exception(type(err), err, sys.exc_info()[2])
            continue

        if isinstance(ret, str):
            print('============', ret, '============')
            continue
        print(f'process success for {cur_success_loop_cnt} ......')
        fout = open(f'cases/case_{cur_success_loop_cnt}.txt', 'w', encoding='utf-8')
        print(f'process success for {cur_success_loop_cnt} ......', file=fout)
        for result, cid1, cname1, cid2, cname2 in ret:
            print('============', cid1, cname1, '||', cid2, cname2, '============', file=fout)
            for idx, (doc, score) in enumerate(result):
                print(f'{idx:03}.', file=fout)
                print(doc['pmid'], score, sep='\t', file=fout)
                print('---', doc['title'].strip(), file=fout)
                # print sentences
                cid2idx = {cid: i for i, cid in enumerate(doc['cids'])}
                mentions = []
                head_set, tail_set = set(), set()
                # 可能不包含全部两个实体 (API 补充)
                if cid1 in cid2idx:
                    for mention in doc['vertexSet'][cid2idx[cid1]]:
                        mentions.append((mention, 0))
                        head_set.add(mention['sent_id'])
                if cid2 in cid2idx:
                    for mention in doc['vertexSet'][cid2idx[cid2]]:
                        mentions.append((mention, 1))
                        tail_set.add(mention['sent_id'])
                star_sents = {}
                for mention, eid in mentions:
                    sid = mention['sent_id']
                    if sid not in star_sents:
                        star_sents[sid] = []
                    star_sents[sid].append((mention['pos'], eid))
                print(f'common_sentences: {str(sorted(list(head_set & tail_set)))}', file=fout)
                for sid, sent in enumerate(doc['sents']):
                    starts, ends = [{}, {}], [{}, {}]
                    if sid in star_sents:
                        print('***', end=' ', file=fout)
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
                    print(f'sent {sid}: {text}', file=fout)
                print(file=fout)
        fout.close()
        cur_success_loop_cnt += 1


if __name__ == '__main__':
    pass
    # main()
