import sys
import csv
import time
import json
import xml.sax
import requests
import traceback
from requests.exceptions import ReadTimeout, ConnectionError


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


class SpellHandler(xml.sax.handler.ContentHandler):
    def __init__(self, query):
        super().__init__()
        self.result = query
        self.current_tag = ''

    def startElement(self, tag, attributes):
        self.current_tag = tag

    def endElement(self, tag):
        self.current_tag = ''

    def characters(self, content):
        if self.current_tag == 'CorrectedQuery' and content != '':
            self.result = content


def repeat_request(url: str, max_time: int = 10):
    for _ in range(max_time):
        try:
            content = requests.get(url, timeout=10).text
            return content
        except (ReadTimeout, ConnectionError):
            print('\ntimeout!', file=sys.stderr)
            time.sleep(1)
        except IOError:
            print('\nother exception timeout!', file=sys.stderr)
            traceback.print_exc()
            time.sleep(1)
    raise RuntimeError('Request Failed!')


def search_term(term: str, db: str = 'mesh', ret_max: int = 100):
    """List of uids"""
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db={db}&term={term}' \
          f'&retmode=json&field=title&retmax={ret_max}'
    url = url.strip().replace(' ', '+')
    cont = json.loads(repeat_request(url))
    try:
        ret = cont['esearchresult']['idlist']
    except KeyError:
        ret = []
        print('key error:', url, cont)
    return ret


def fetch_uid(uids: list, db: str = 'mesh'):
    uid_str = ','.join(uids)
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db={db}&id={uid_str}&retmode=json'
    cont = repeat_request(url)
    return cont


def summary_uids(uids: list, db: str = 'mesh'):
    uid_str = ','.join(uids)
    """uid -> tuple[description, entry_terms (synonyms), link_entities, MeSH_ID]"""
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db={db}&id={uid_str}&retmode=json'
    cont = json.loads(repeat_request(url))['result']
    if len(cont['uids']) != len(uids):
        print(f'Summary Failed: {uid_str} result: {cont}', file=sys.stderr)
        return {}
    ret = {}
    for uid in cont['uids']:
        entity: dict = cont[uid]
        entity.setdefault('ds_scopenote', '')
        entity.setdefault('ds_meshterms', [])
        entity.setdefault('ds_idxlinks', [])
        entity.setdefault('ds_meshui', 'UNK')
        mesh_id = entity['ds_meshui']
        if mesh_id != 'UNK':
            # NCBI ID 与 MeSH ID 之间可直接变换
            assert str(ord(mesh_id[0])) + mesh_id[1:] == uid
        ret[uid] = entity['ds_scopenote'], entity['ds_meshterms'], entity['ds_idxlinks'], mesh_id
    return ret


def spell_term(term: str, db: str = 'mesh'):
    """拼写纠错，编辑距离较短时效果较好"""
    url = f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/espell.fcgi?db={db}&term={term}'
    url = url.strip().replace(' ', '+')
    cont = repeat_request(url)
    handler = SpellHandler(term)
    xml.sax.parseString(cont, handler)
    return handler.result


def get_pmids(pmids: list, concepts: list = None):
    if len(pmids) == 0:
        return {}
    if concepts is None:
        concepts = ['chemical', 'disease', 'gene']
    concept_str = ','.join(concepts)
    pmid_str = ','.join(pmids)
    url = f'https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocjson?' \
          f'pmids={pmid_str}&concepts={concept_str}'
    cont = repeat_request(url)
    ret = {}
    for js in cont.split('\n'):
        if len(js.strip()) == 0:
            continue
        doc = json.loads(js)
        passages = doc['passages']
        title = ''
        texts = []
        annotations = []
        if len(passages) > 0:
            title = passages[0]['text']
        for section in passages:
            texts.append(section['text'])
            annotations += section['annotations']
        ret[doc['id']] = {
            'pmid': doc['id'],
            'title': title,
            'text': ' '.join(texts),
            'entities': annotations
        }
    return ret


def search_get_pubmed(entities: list, pmid_filter: set = None, ent_pair: tuple = None, ret_max: int = 100):
    id_list = search_term(' '.join(entities), db='pubmed', ret_max=ret_max)
    if pmid_filter is not None:
        # 过滤已知包含正例的文章
        id_list = [pid for pid in id_list if pid not in pmid_filter]
    ret = get_pmids(id_list)
    if ent_pair is not None:
        # 过滤不包含这两个实体的文章
        tmp_ret = {}
        for did, info in ret.items():
            h_in, t_in = False, False
            for entity in info['entities']:
                eid = entity['infons']['identifier']
                if eid is None:
                    eid = '-'
                if eid.startswith('MESH:'):
                    eid = eid[5:]
                if eid == ent_pair[0]:
                    h_in = True
                if eid == ent_pair[1]:
                    t_in = True
            if h_in and t_in:
                tmp_ret[did] = info
        ret = tmp_ret
    return ret


def print_json(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


def is_mesh_id(cid: str):
    return cid[0].isalpha() and cid[1:].isdigit()


def main1():
    """生成 CTD 关系"""
    rel2id = {'therapeutic': 1, 'marker/mechanism': 0}
    fin = open('CTDRED/CTD_chemicals_diseases.csv', 'r', encoding='utf-8')
    reader = csv.reader(fin)
    rel_set, rel_null_set = {}, {}
    for line in reader:
        if len(line) != 10 or len(line) > 0 and line[0].startswith('#'):
            continue
        chemical_id, disease_id, rel, pmids = line[1], line[4], line[5], line[9]
        assert rel in {'', 'therapeutic', 'marker/mechanism'}
        if chemical_id.startswith('MESH:'):
            chemical_id = chemical_id[5:]
        assert is_mesh_id(chemical_id)
        if disease_id.startswith('MESH:'):
            disease_id = disease_id[5:]
        if disease_id.startswith('OMIM'):
            # 直接忽略
            continue
        assert is_mesh_id(disease_id)
        if rel == '':
            if (chemical_id, disease_id) not in rel_null_set:
                rel_null_set[(chemical_id, disease_id)] = set()
            for pmid in pmids.split('|'):
                rel_null_set[(chemical_id, disease_id)].add(pmid)
        else:
            if (chemical_id, disease_id) not in rel_set:
                rel_set[(chemical_id, disease_id)] = set()
            for pmid in pmids.split('|'):
                rel_set[(chemical_id, disease_id)].add((rel2id[rel], pmid))
    fin.close()
    # 98814 2725363
    print(len(rel_set), len(rel_null_set))
    save_json([list(key) + list(value) for key, value in rel_set.items()], 'CTDRED/ctd_overall_rel.json')
    save_json([list(key) + list(value) for key, value in rel_null_set.items()], 'CTDRED/ctd_overall_null_rel.json')
    rel_set = {(li[0], li[1]): [tuple(t) for t in li[2:]] for li in load_json('CTDRED/ctd_overall_rel.json')}
    doc2labels = {}
    for key, val in rel_set.items():
        hid, tid = key
        for lab in val:
            rel, pmid = lab
            if pmid not in doc2labels:
                doc2labels[pmid] = []
            doc2labels[pmid].append((hid, tid, rel))
    # for key, val in rel_null_set.items():
    #     hid, tid = key
    #     for pmid in val:
    #         if pmid not in doc2labels:
    #             doc2labels[pmid] = []
    #         doc2labels[pmid].append((hid, tid, -1))
    save_json(doc2labels, 'CTDRED/ctd_doc_to_labels.json')


def pubtator_to_docred(doc, labels):
    offsets = []
    eid2offsets = {}
    for entity in doc['entities']:
        if entity['infons']['type'] not in ('Chemical', 'Disease'):
            continue
        cid = entity['infons']['identifier']
        if cid is None:
            continue
        name = entity['text']
        for loc in entity['locations']:
            off, le = loc['offset'], loc['length']
            offsets.append((off, off + le))
            if cid.startswith('MESH:'):
                cid = cid[5:]
            if cid not in eid2offsets:
                eid2offsets[cid] = []
            eid2offsets[cid].append((off, off + le, entity['infons']['type'], name))

    title, text = doc['title'], doc['text']
    offset_to_sid = [-1000000] * len(text)

    assert text[len(title)] == ' '
    text = text[(len(title) + 1):]

    accu_sent_len, doc_sent = [], []

    # process the title
    title_tokens = title.split(' ')
    # 包含最后 1 空格
    cur_sent_len = sum(len(item) for item in title_tokens) + len(title_tokens)
    for k in range(0, cur_sent_len - 1):
        offset_to_sid[k] = len(doc_sent)
    accu_sent_len.append(cur_sent_len)
    doc_sent.append(title_tokens)
    cur_sent_len_total = cur_sent_len

    # generate sentences
    tokens = text.split(' ')
    cur_sent = []
    for tid, token in enumerate(tokens):
        cur_sent.append(token)
        cur_sent_len_total += len(token) + 1
        if token.endswith('.'):
            if len(token) == 2 or len(token) == 3 and token[0] == '(':
                continue

            middle_flag = False
            for start, end in offsets:
                if start < cur_sent_len_total <= end:
                    middle_flag = True
            if middle_flag:
                continue

            # generate a sentence
            accu_len = accu_sent_len[-1]
            # 此处包含最后一空格
            cur_sent_len = sum(len(item) for item in cur_sent) + len(cur_sent)
            for k in range(accu_len, accu_len + cur_sent_len - 1):
                offset_to_sid[k] = len(doc_sent)
            accu_len += cur_sent_len
            accu_sent_len.append(accu_len)
            doc_sent.append(cur_sent)
            cur_sent = []
    if len(cur_sent) > 0:
        # last sentence
        accu_len = accu_sent_len[-1]
        cur_sent_len = sum(len(item) for item in cur_sent) + len(cur_sent)
        for k in range(accu_len, accu_len + cur_sent_len - 1):
            offset_to_sid[k] = len(doc_sent)
        accu_len += cur_sent_len
        accu_sent_len.append(accu_len)
        doc_sent.append(cur_sent)

    vertex_set = []
    mesh2id = {}
    cids = []
    for eid, mentions in eid2offsets.items():
        cur_entity = []
        assert eid not in mesh2id
        mesh2id[eid] = len(mesh2id)
        cids.append(eid)
        for start, end, typ, name in mentions:
            assert start < end
            sid = offset_to_sid[start]
            assert 0 <= sid == offset_to_sid[end - 1] and sid < len(doc_sent) and \
                   len(accu_sent_len) == len(doc_sent)
            prev_sent_accu = accu_sent_len[sid - 1] if sid > 0 else 0
            start -= prev_sent_accu
            end -= prev_sent_accu
            cur_sent = doc_sent[sid]
            left_pos, right_pos = -1, -1
            begin_pos, end_pos = -1, -1
            for k in range(0, len(cur_sent)):
                left_pos = right_pos + 1
                right_pos += len(cur_sent[k]) + 1
                if left_pos <= start < right_pos:
                    begin_pos = k
                if begin_pos >= 0 and left_pos < end <= right_pos:
                    end_pos = k + 1
                    break
            assert 0 <= begin_pos < end_pos <= len(cur_sent)
            cur_entity.append({
                'name': name,
                'sent_id': sid,
                'pos': [begin_pos, end_pos],
                'type': typ,
            })
        vertex_set.append(cur_entity)

    id2rel = ['chem_disease_marker/mechanism', 'chem_disease_therapeutic']
    label_set = []
    for hid, tid, rel in labels:
        if not (hid in mesh2id and tid in mesh2id):
            continue
        h, t = mesh2id[hid], mesh2id[tid]
        assert vertex_set[h][0]['type'] == 'Chemical' and vertex_set[t][0]['type'] == 'Disease'
        label_set.append({
            'h': h,
            't': t,
            'r': id2rel[rel],
        })

    return {
        'pmid': doc['pmid'],
        'cids': cids,
        'vertexSet': vertex_set,
        'title': title,
        'sents': doc_sent,
        'labels': label_set
    }


def main2():
    """扩充现有的 CTD 数据集"""
    rel_set = {(li[0], li[1]): [tuple(t) for t in li[2:]] for li in load_json('CTDRED/ctd_overall_rel.json')}
    rel_null_set = {(li[0], li[1]): li[2:] for li in load_json('CTDRED/ctd_overall_null_rel.json')}
    doc2labels = {key: tuple(value) for key, value in load_json('CTDRED/ctd_doc_to_labels.json').items()}
    rel_exist_set = {}
    in_data_pmids = set()
    for part in ['test', 'dev']:
        data = load_json(f'CTDRED/negative_{part}.json')
        for doc in data:
            in_data_pmids.add(doc['pmid'])
    for part in ('test', 'dev', 'train_mixed'):
        data = load_json(f'CTDRED/{part}.json')
        for doc in data:
            assert doc['pmid'] not in in_data_pmids
            in_data_pmids.add(doc['pmid'])
    err_log = open('CTDRED/err_log.txt', 'w', encoding='utf-8')
    for part in ['train_mixed']:
        data = load_json(f'CTDRED/{part}.json')
        pair_sets = {}
        for doc in data:
            cids = doc['cids']
            entities = doc['vertexSet']
            for lab in doc['labels']:
                h, t = lab['h'], lab['t']
                if not (is_mesh_id(cids[h]) and is_mesh_id(cids[t]) and entities[h][0]['type'] == 'Chemical'
                        and entities[t][0]['type'] == 'Disease'):
                    continue
                pair_sets[(cids[h], cids[t])] = (
                    max([n['name'] for n in entities[h]], key=lambda x: len(x)),
                    max([n['name'] for n in entities[t]], key=lambda x: len(x))
                )
                if (cids[h], cids[t]) not in rel_exist_set:
                    rel_exist_set[(cids[h], cids[t])] = []
                rel_exist_set[(cids[h], cids[t])].append(str(doc['pmid']))
        negative_doc_ids = {}
        pair_sets_len, cur_pair_cnt = len(pair_sets), 0
        for pair, names in pair_sets.items():
            filter_set = set()
            if pair in rel_set:
                for pid in rel_set[pair]:
                    filter_set.add(pid[1])
            if pair in rel_null_set:
                for pid in rel_null_set[pair]:
                    filter_set.add(pid)
            if pair in rel_exist_set:
                for pid in rel_exist_set[pair]:
                    filter_set.add(pid)
            filter_set |= in_data_pmids
            docs = search_get_pubmed(list(names), pmid_filter=filter_set, ent_pair=pair)
            for did, val in docs.items():
                try:
                    negative_doc_ids[did] = pubtator_to_docred(val, doc2labels[did] if did in doc2labels else [])
                except Exception as err:
                    err_log.write(did + '\n')
                    traceback.print_exception(type(err), err, sys.exc_info()[2], file=err_log)
                    err_log.write('============\n')
                in_data_pmids.add(did)
            cur_pair_cnt += 1
            print(f'{part} complete {cur_pair_cnt:5}/{pair_sets_len:5} documents: {len(negative_doc_ids):7}', end='\r')
        print('\ncomplete part:', part)
        save_json(list(negative_doc_ids.values()), f'CTDRED/negative_{part}.json')
    err_log.close()


def add_null_labels(binary=False):
    """
    添加 CTD 中的空关系
    train_mixed 38595 16685 -> 15288
    dev 14568 8863 -> 7992
    test 14987 9151 -> 8184
    negative_train_mixed 1655 1727 -> 1655
    negative_dev 1324 1371 -> 1324
    negative_test 2393 2474 -> 2393
    """
    doc2labels = load_json('CTDRED/ctd_doc_to_labels.json')
    id2rel = ['chem_disease_marker/mechanism', 'chem_disease_therapeutic', 'UNK']
    for path in ['train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test']:
        data = load_json(f'CTDRED/{path}.json')
        total_instances = sum(len(doc['labels']) for doc in data)
        for doc in data:
            # eliminate genes
            cids, vertex_set, labels = doc['cids'], doc['vertexSet'], doc['labels']
            assert len(cids) == len(vertex_set)
            new_cids, new_vertex_set, new_labels = [], [], set()
            idx_map = {}
            for idx in range(len(cids)):
                cid, entity = cids[idx], vertex_set[idx]
                if entity[0]['type'] in ('Chemical', 'Disease'):
                    idx_map[idx] = len(idx_map)
                    new_cids.append(cid)
                    new_vertex_set.append(entity)

            pmid = str(doc['pmid'])
            new_cid2idx = {cid: i for i, cid in enumerate(new_cids)}
            extra_labels = doc2labels[pmid] if pmid in doc2labels else []

            for lab in labels:
                if lab['r'] in ('chem_disease_marker/mechanism', 'chem_disease_therapeutic'):
                    if lab['h'] not in idx_map or lab['t'] not in idx_map:
                        continue
                    new_h, new_t = idx_map[lab['h']], idx_map[lab['t']]
                    new_labels.add((new_h, new_t, lab['r']))

            for hid, tid, rel in extra_labels:
                if not (hid in new_cid2idx and tid in new_cid2idx):
                    continue
                h, t = new_cid2idx[hid], new_cid2idx[tid]
                assert new_vertex_set[h][0]['type'] == 'Chemical' and new_vertex_set[t][0]['type'] == 'Disease'
                new_labels.add((h, t, id2rel[rel]))
            label_list = []
            if binary:
                positive_pairs = set()
                for h, t, rel in new_labels:
                    assert new_vertex_set[h][0]['type'] == 'Chemical' and new_vertex_set[t][0]['type'] == 'Disease'
                    positive_pairs.add((h, t))
                for h, t in positive_pairs:
                    label_list.append({
                        'h': h,
                        't': t,
                        'r': 'Pos'
                    })
                entity_num = len(new_vertex_set)
                for h in range(entity_num):
                    for t in range(entity_num):
                        if not (new_vertex_set[h][0]['type'] == 'Chemical' and
                                new_vertex_set[t][0]['type'] == 'Disease'):
                            continue
                        if (h, t) in positive_pairs:
                            continue
                        label_list.append({
                            'h': h,
                            't': t,
                            'r': 'NA'
                        })
            else:
                for h, t, rel in new_labels:
                    assert new_vertex_set[h][0]['type'] == 'Chemical' and new_vertex_set[t][0]['type'] == 'Disease'
                    label_list.append({
                        'h': h,
                        't': t,
                        'r': rel
                    })
            doc['cids'] = new_cids
            doc['vertexSet'] = new_vertex_set
            doc['labels'] = label_list
        save_json(data, f'CTDRED/{path}_{"binary2" if binary else "null"}.json')
        after_instances = sum(len(doc['labels']) for doc in data)
        print(path, total_instances, after_instances)


if __name__ == '__main__':
    # 扩充数据集
    # result = search_get_pubmed(['Naloxone', 'clonidine'])
    # pubtator_to_docred(list(result.values())[0])
    # exit()
    # main1()
    # main2()
    # add_null_labels()
    # add_null_labels(binary=True)

    # data3 = load_json('CTDRED/train_mixed_binary3.json')
    # data4 = load_json('CTDRED/train_mixed_binary4.json')
    # print(len(data3), len(data4))
    # print(data3[0]['labels'], data3[0]['pmid'])
    # print(data4[0]['labels'], data4[0]['pmid'])
    # for doc3, doc4 in zip(data3, data4):
    #     set3, set4 = set(), set()
    #     for lab in doc3['labels']:
    #         if lab['r'] == 'Pos':
    #             set3.add((lab['h'], lab['t']))
    #     for lab in doc4['labels']:
    #         set4.add((lab['h'], lab['t']))
    #     assert len(set3 & set4) == len(set3) == len(set4)
    # exit()

    # for part in ('train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test'):
    #     data = load_json(f'CTDRED/{part}_binary_pos.json')
    #     data2 = load_json(f'CTDRED/{part}_binary_na.json')
    #     assert len(data) == len(data2)
    #     title_set = {}
    #     cnt = 0
    #     for doc, doc2 in zip(data, data2):
    #         assert '-' not in doc['cids']
    #         assert '-' not in doc2['cids']
    #         assert doc['pmid'] == doc2['pmid']
    #         assert doc['title'] == doc2['title']
    #         title = doc['title']
    #         if title not in title_set:
    #             title_set[title] = 1
    #         else:
    #             cnt += 1
    #             idx = title_set[title]
    #             doc['title'] = title + ' ' + str(idx)
    #             doc2['title'] = title + ' ' + str(idx)
    #             title_set[title] = idx + 1
    #             # data[cur_idx] = doc
    #             # data2[cur_idx] = doc2
    #     print(part, cnt)
    #     if cnt > 0:
    #         raise RuntimeError
    #         save_json(data, f'CTDRED/{part}_binary_pos.json')
    #         save_json(data2, f'CTDRED/{part}_binary_na.json')
    # exit()

    # import numpy as np
    # for part in ['train_mixed', 'dev', 'test']:
    #     title_score = load_json(f'CTDRED/rank_result_ctdred_binary/negative_{part}_binary_pos_title.json')
    #     title2idx = {title: i for i, title in enumerate(title_score)}
    #     scores = np.load(f'CTDRED/rank_result_ctdred_binary/negative_{part}_binary_pos_score.npy')
    #     doc_cnt, pad = scores.shape
    #     title_fixed = load_json(f'CTDRED/rank_result_ctdred_binary_fix/negative_{part}_binary_pos_title.json')
    #     title2idx_fixed = {title: i for i, title in enumerate(title_fixed)}
    #     score_fixed = np.load(f'CTDRED/rank_result_ctdred_binary_fix/negative_{part}_binary_pos_score.npy')
    #     fix_cnt, pad_fix = score_fixed.shape
    #     assert pad <= pad_fix
    #     for t in title_fixed:
    #         idx, idx_fix = title2idx[t], title2idx_fixed[t]
    #         scores[idx] = score_fixed[idx_fix, :pad]
    #     np.save(f'CTDRED/rank_result_ctdred_binary/negative_{part}_binary_pos_score.npy', scores)
    # exit()

    for part in ('train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test'):
        title1 = load_json(f'CTDRED/rank_result_ctdred_binary_o/{part}_binary_pos_title.json')
        title2 = load_json(f'CTDRED/ctd_binary_denoise_n15_inter/{part}_binary_pos_title.json')
        assert title1 == title2
    exit()

    '''
    binary3 包含 NA, binary4 不包含 NA
    存在无标签的文档
    '''
    for part in ('train_mixed', 'dev', 'test'):
        data3 = load_json(f'CTDRED/{part}_binary3.json')
        data4 = load_json(f'CTDRED/{part}_binary_na.json')
        data5 = load_json(f'CTDRED/{part}_binary_pos.json')
        print(len(data3), len(data4), len(data5))
        # 3467 5603 5603
        # 3878 5341 5341
        # 3960 5371 5371
        assert len(data3) < len(data4) == len(data5)
        data4 = [item for item in data4 if len([lab for lab in item['labels'] if lab['r'] != 'NA']) > 0]
        data5 = [item for item in data5 if len(item['labels']) > 0]
        assert len(data3) == len(data4) == len(data5)
        for doc3, doc4, doc5 in zip(data3, data4, data5):
            assert doc3['pmid'] == doc4['pmid'] == doc5['pmid']
            assert doc3['vertexSet'] == doc4['vertexSet'] == doc5['vertexSet']
            assert doc3['sents'] == doc4['sents'] == doc5['sents']
            assert doc3['cids'] == doc4['cids'] == doc5['cids']
            assert doc3['title'] == doc4['title'] == doc5['title']
            set3, set4, set5 = set(), set(), set()
            for lab in doc3['labels']:
                set3.add((lab['h'], lab['t'], lab['r']))
            for lab in doc4['labels']:
                set4.add((lab['h'], lab['t'], lab['r']))
            assert len(set3) == len(set4) == len(set3 & set4)
            set4.clear()
            for lab in doc4['labels']:
                if lab['r'] != 'NA':
                    set4.add((lab['h'], lab['t'], lab['r']))
            for lab in doc5['labels']:
                set5.add((lab['h'], lab['t'], lab['r']))
            assert len(set4) == len(set5) == len(set4 & set5)
    exit()

    doc2labels = load_json('CTDRED/ctd_doc_to_labels.json')
    id2rel = ['chem_disease_marker/mechanism', 'chem_disease_therapeutic']
    for part in ('train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test'):
        data = load_json(f'CTDRED/{part}.json')
        total_instances = sum(len(doc['labels']) for doc in data)
        new_data = []
        cnt = 0
        for doc in data:
            # eliminate genes
            cids, vertex_set, labels = doc['cids'], doc['vertexSet'], doc['labels']
            assert len(cids) == len(vertex_set)
            new_cids, new_vertex_set, new_labels = [], [], set()
            idx_map = {}
            for idx in range(len(cids)):
                cid, entity = cids[idx], vertex_set[idx]
                if entity[0]['type'] in ('Chemical', 'Disease') and cids[idx] != '-':
                    idx_map[idx] = len(idx_map)
                    new_cids.append(cid)
                    new_vertex_set.append(entity)

            pmid = str(doc['pmid'])
            new_cid2idx = {cid: i for i, cid in enumerate(new_cids)}

            for lab in labels:
                if lab['r'] in ('chem_disease_marker/mechanism', 'chem_disease_therapeutic'):
                    if lab['h'] not in idx_map or lab['t'] not in idx_map:
                        continue
                    new_h, new_t = idx_map[lab['h']], idx_map[lab['t']]
                    new_labels.add((new_h, new_t, lab['r']))

            extra_labels = doc2labels[pmid] if pmid in doc2labels else []

            for hid, tid, rel in extra_labels:
                if not (hid in new_cid2idx and tid in new_cid2idx):
                    continue
                h, t = new_cid2idx[hid], new_cid2idx[tid]
                assert new_vertex_set[h][0]['type'] == 'Chemical' and new_vertex_set[t][0]['type'] == 'Disease'
                assert rel in (0, 1)
                new_labels.add((h, t, id2rel[rel]))

            label_list = []
            positive_pairs = set()
            for h, t, rel in new_labels:
                assert new_vertex_set[h][0]['type'] == 'Chemical' and new_vertex_set[t][0]['type'] == 'Disease'
                positive_pairs.add((h, t))
            positive_pairs = sorted(list(positive_pairs))
            for h, t in positive_pairs:
                label_list.append({
                    'h': h,
                    't': t,
                    'r': 'Pos'
                })
            # entity_num = len(new_vertex_set)
            # for h in range(entity_num):
            #     for t in range(entity_num):
            #         if not (new_vertex_set[h][0]['type'] == 'Chemical' and
            #                 new_vertex_set[t][0]['type'] == 'Disease'):
            #             continue
            #         if (h, t) in positive_pairs:
            #             continue
            #         label_list.append({
            #             'h': h,
            #             't': t,
            #             'r': 'NA'
            #         })
            doc['cids'] = new_cids
            doc['vertexSet'] = new_vertex_set
            doc['labels'] = label_list

            has_c, has_d = False, False
            for entity in new_vertex_set:
                has_c |= entity[0]['type'] == 'Chemical'
                has_d |= entity[0]['type'] == 'Disease'
            if has_c and has_d:
                new_data.append(doc)
                cnt += 1
        print(len(data), len(new_data), cnt)
        save_json(new_data, f'CTDRED/{part}_binary_pos.json')
        after_instances = sum(len(doc['labels']) for doc in new_data)
        print(part, total_instances, after_instances)
    exit()

    for part in ('train_mixed', 'dev', 'test'):
        data_base = load_json(f'CTDRED/{part}.json')
        data_binary = load_json(f'CTDRED/{part}_binary2.json')
        unk_cnt = 0
        for base, binary in zip(data_base, data_binary):
            assert base['pmid'] == binary['pmid']
            new_labels = set()
            new_cids = binary['cids']
            for lab in binary['labels']:
                r = lab['r']
                assert r in ('Pos', 'NA')
                if r != 'NA':
                    h, t = lab['h'], lab['t']
                    new_labels.add((new_cids[h], new_cids[t]))

            cids = base['cids']
            entities = base['vertexSet']
            base_labels = set()
            for lab in base['labels']:
                h, t, r = lab['h'], lab['t'], lab['r']
                if entities[h][0]['type'] == 'Chemical' and entities[t][0]['type'] == 'Disease':
                    base_labels.add((cids[h], cids[t]))

            intersect = base_labels & new_labels
            assert len(intersect) == len(base_labels)
            unk_cnt += len(new_labels) - len(base_labels)
        print(part, unk_cnt)
    exit()
