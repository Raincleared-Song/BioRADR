import os
import re
import random
from tqdm import tqdm
from ncbi_api import baidu_translate, fetch_uids, is_mesh_id, pubtator_to_docred, search_get_pubmed
from search_db import init_db, get_documents_by_pmids
from search_utils import save_json, load_json, adaptive_load, fix_ner_by_search


def gen_pair_to_docs():
    pair_to_docs = {}
    for part in ('train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test',
                 'negative_train_extra', 'negative_dev_extra', 'negative_test_extra', 'ctd_extra', 'extra_batch'):
        data_iter = adaptive_load(f'../project-1/CTDRED/{part}_binary_pos')
        cur_cnt = 0
        for doc in data_iter:
            cids = doc['cids']
            entities = doc['vertexSet']
            entity_num = len(entities)
            label_set = set()
            for lab in doc['labels']:
                assert lab['r'] == 'Pos'
                label_set.add((lab['h'], lab['t']))
                assert entities[lab['h']][0]['type'] == 'Chemical' and entities[lab['t']][0]['type'] == 'Disease'
            for i in range(entity_num):
                for j in range(entity_num):
                    if not (entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease'):
                        continue
                    pair = cids[i] + '&' + cids[j]
                    if pair not in pair_to_docs:
                        pair_to_docs[pair] = ([], [])
                    pair_to_docs[pair][int((i, j) in label_set)].append(int(doc['pmid']))
            cur_cnt += 1
            print(f'{part} processed: {cur_cnt:07}', end='\r')
        del data_iter
        print()
    save_json(pair_to_docs, '../project-1/CTDRED/pair_to_docs.json')


def get_sample_pairs():
    init_db()
    random.seed(66)
    pair_to_docs = load_json('CTDRED/pair_to_docs.json')
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    mesh_id_to_counts = load_json('CTDRED/mesh_id_to_counts.json')
    lower_bound, upper_bound, lab_lower, lab_upper = 50, 1e8, -1, 2
    lower_ent_bound, upper_ent_bound = 0, 100000
    cnt_thred, cnt_lab_thred, doc_cnt = 0, 0, 0
    candidate_pairs_in, candidate_pairs_out = [], []

    ctd_relations, ctd_entities = set(), set()
    ctd_doc_to_relations = load_json('../project-1/CTDRED/ctd_doc_to_labels_complete.json')
    for _, value in ctd_doc_to_relations.items():
        for h, t, _ in value:
            assert h != '-' and t != '-'
            pair = h + '&' + t
            ctd_relations.add(pair)
    ctd_doc_to_relations = load_json('../project-1/CTDRED/ctd_doc_to_labels.json')
    for _, value in ctd_doc_to_relations.items():
        for h, t, _ in value:
            assert h != '-' and t != '-'
            ctd_entities.add(h)
            ctd_entities.add(t)

    # 选实体对和文章
    print(len(pair_to_docs))
    filter_list = ['D002241', 'D005947', 'D002244', 'D007501', 'D002248', 'D011041', 'D003300', 'D005483', 'D007022',
                   'D000073893', 'D014867', 'D010100', 'D010758', 'D009369', 'D003643', 'D003681', 'D006261',
                   'D009325', 'D063646', 'D004342', 'D006973', 'D009503', 'D000740', 'D013213']
    for pair, val in pair_to_docs.items():
        cur_doc_num = len(val[0] + val[1])
        h, t = pair.split('&')
        h_occur, t_occur = mesh_id_to_counts[h], mesh_id_to_counts[t]
        if upper_bound > cur_doc_num >= lower_bound and upper_ent_bound > max(h_occur, t_occur) >= lower_ent_bound:
            if h not in mesh_id_to_name or t not in mesh_id_to_name or h not in ctd_entities or t not in ctd_entities:
                continue
            if h in filter_list or t in filter_list or not is_mesh_id(h) or not is_mesh_id(t):
                continue
            t_name = mesh_id_to_name[t].lower()
            if 'disease' in t_name or 'pain' in t_name or 'disorder' in t_name or 'failure' in t_name \
                    or 'disorders' in t_name or 'loss' in t_name or 'gain' in t_name or 'insufficiency' in t_name\
                    or 'heart' in t_name or 'injury' in t_name or 'ulcer' in t_name or 'neoplasms' in t_name \
                    or 'hypertension' in t_name or 'infections' in t_name:
                continue
            cnt_thred += 1
            if lab_upper * cur_doc_num > len(val[1]) >= lab_lower * cur_doc_num:
                cnt_lab_thred += 1
                doc_cnt += cur_doc_num
                if pair in ctd_relations:
                    candidate_pairs_in.append((h, t))
                else:
                    candidate_pairs_out.append((h, t))

    print(len(candidate_pairs_in), len(candidate_pairs_out))
    # half in ctd and half not in ctd
    candidate_pairs = random.sample(candidate_pairs_in, 25) + random.sample(candidate_pairs_out, 25)
    candidate_entities = set()
    for h, t in candidate_entities:
        candidate_entities.add(h)
        candidate_entities.add(t)

    # replace common entities
    def part_replace(replace_list):
        for idx in replace_list:
            if idx < 25:
                chosen = random.choice(candidate_pairs_in)
                while chosen[0] in candidate_entities or chosen[1] in candidate_entities:
                    chosen = random.choice(candidate_pairs_in)
            else:
                chosen = random.choice(candidate_pairs_out)
                while chosen[0] in candidate_entities or chosen[1] in candidate_entities:
                    chosen = random.choice(candidate_pairs_out)
            candidate_pairs[idx] = chosen
            candidate_entities.add(chosen[0])
            candidate_entities.add(chosen[1])

    to_inspect = set(range(50))
    while len(to_inspect) > 0:
        # filter Inorganic Chemicals, Pathological Conditions, Signs and Symptoms,
        for pid in tqdm(list(to_inspect), desc='replace inspect'):
            h, t = candidate_pairs[pid]
            if check_pair_by_category(h, t):
                to_inspect.discard(pid)
        part_replace(to_inspect)
    part_replace([30, 46])

    fout = open('manual/candidates2.txt', 'w')
    for pid, pair in enumerate(candidate_pairs):
        h, t = pair
        val = pair_to_docs[h + '&' + t]
        val = val[0] + val[1]
        # sample if more than 100 documents
        if len(val) > 100:
            val = random.sample(val, 100)
        fout.write(f'{pid}\t{h}\t{t}\t{len(val)}\t{mesh_id_to_name[h]}\t{mesh_id_to_counts[h]}'
                   f'\t{mesh_id_to_name[t]}\t{mesh_id_to_counts[t]}\n')
    fout.close()
    print(len(pair_to_docs), cnt_thred, cnt_lab_thred, doc_cnt)


def check_pair_by_category(head, tail):
    """实体处在至少一棵分类树末端, 且不是特定类型"""
    head, tail = str(ord(head[0])) + head[1:], str(ord(tail[0])) + tail[1:]
    cont = fetch_uids([head, tail])
    if 'Inorganic Chemicals' in cont or 'Pathological Conditions, Signs and Symptoms' in cont:
        return False
    hs, ts = cont.find('1: '), cont.find('2: ')
    if hs == -1 or ts == -1:
        return False
    h_cont, t_cont = cont[hs:ts].split('\n'), cont[ts:].split('\n')
    try:
        assert h_cont[0][2] == ' ' and t_cont[0][2] == ' '
    except AssertionError as err:
        print(head, tail)
        print(h_cont)
        print(t_cont)
        raise err
    h_name, t_name = h_cont[0][3:], t_cont[0][3:]
    h_pass, t_pass = False, False
    for tid, token in enumerate(h_cont):
        if h_name in token and '        ' in token:
            if h_cont[tid + 1] == '' or h_cont[tid + 1].strip().startswith(h_name):
                h_pass = True
                break
    for tid, token in enumerate(t_cont):
        if t_name in token and '        ' in token:
            if t_cont[tid + 1] == '' or t_cont[tid + 1].strip().startswith(t_name):
                t_pass = True
                break
    return h_pass and t_pass


def batch_translate_write(cur_mark_sents, cur_raw_sents, fout_data, fout_log, pmid):
    # 批量翻译
    zh_translation, load_length = None, 0
    for _ in range(10):
        try:
            zh_translation = ['' for _ in range(len(cur_raw_sents))]
            # zh_translation, load_length = baidu_translate(cur_raw_sents)
            break
        except Exception as err:
            print(err)
    if zh_translation is None:
        raise RuntimeError('baidu translation failure!')
    if len(zh_translation) != len(cur_raw_sents):
        to_translate = '\n'.join(cur_raw_sents)
        fout_log.write(f'{pmid}\n')
        fout_log.write(f'{to_translate}\n\n')
        zh_translation = ['' for _ in range(len(cur_raw_sents))]
    assert len(zh_translation) == len(cur_raw_sents) == len(cur_mark_sents)
    for ssid in range(len(cur_raw_sents)):
        ss, mm = cur_mark_sents[ssid]
        zh_sent = zh_translation[ssid]
        fout_data.write(f'{"*** " if mm else ""}SENTENCE {ssid}: {ss}\n')
        fout_data.write(f'{"*** " if mm else ""}句子 {ssid}: {zh_sent}\n\n')
    return load_length, len(cur_raw_sents)


def get_single_pair_document(task_name: str, documents: list, ent_pair: tuple, names: tuple,
                             counts: tuple, err_log, pmc: bool = False):
    total_byte_length = 0
    f_data = open(f'manual/{task_name}.txt', 'w', encoding='utf-8')
    f_answer = open(f'manual/{task_name}_ans.txt', 'w')
    h, t = ent_pair
    pmid_key = 'pmsid' if pmc else 'pmid'
    f_data.write('[head_MESH]\t[tail_MESH]\t[num_of_docs]\t[head_name]\thead_occurrence\t[tail_name]'
                 '\ttail_occurrence\n')
    f_data.write(f'[{h}]\t[{t}]\t[{len(documents)}]\t[{names[0]}]\t{counts[0]}\t[{names[1]}]\t{counts[1]}\n\n')
    p_bar = tqdm(len(documents), desc=task_name)
    for did, doc in enumerate(documents):
        f_answer.write(f'{doc[pmid_key]},\n')
        f_data.write(f'================================= doc{did:03} begin =================================\n\n')
        f_data.write('[head_MESH]\t[tail_MESH]\t[num_of_docs]\t[head_name]\t[tail_name]\n')
        f_data.write(f'[{h}]\t[{t}]\t[{len(documents)}]\t[{names[0]}]\t[{names[1]}]\n\n')
        f_data.write(f'pmid: {doc[pmid_key]} sentences:\n\n')
        # write sentences
        cid2idx = {cid: i for i, cid in enumerate(doc['cids'])}
        mentions = []
        # 可能不包含全部两个实体 (API 补充)
        if h in cid2idx:
            mentions += [(mention, 0) for mention in doc['vertexSet'][cid2idx[h]]]
        if t in cid2idx:
            mentions += [(mention, 1) for mention in doc['vertexSet'][cid2idx[t]]]
        star_sents = {}
        for mention, eid in mentions:
            sid = mention['sent_id']
            if sid not in star_sents:
                star_sents[sid] = []
            star_sents[sid].append((mention['pos'], eid))

        cc_mark_sents, cc_raw_sents, cc_raw_sents_len, written_sent_num = [], [], 0, 0
        for sid, sent in enumerate(doc['sents']):
            starts, ends = [{}, {}], [{}, {}]
            is_star_sent = False
            if sid in star_sents:
                is_star_sent = True
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
            if len(tokens) <= 0:
                written_sent_num += 1
                p_bar.update()
                continue
            text = ' '.join(tokens)
            raw_tokens = [t.strip() for t in sent if t.strip() != '']
            raw_sent = ' '.join(raw_tokens)

            if cc_raw_sents_len + len(raw_sent) >= 6000:
                trans_len, written_num = batch_translate_write(
                    cc_mark_sents, cc_raw_sents, f_data, err_log, doc[pmid_key])
                total_byte_length += trans_len
                written_sent_num += written_num
                cc_mark_sents, cc_raw_sents, cc_raw_sents_len = [], [], 0

            cc_mark_sents.append((text, is_star_sent))
            cc_raw_sents.append(raw_sent)
            cc_raw_sents_len += len(raw_sent) + 1

        trans_len, written_num = batch_translate_write(cc_mark_sents, cc_raw_sents, f_data, err_log, doc[pmid_key])
        total_byte_length += trans_len
        written_sent_num += written_num
        assert written_sent_num == len(doc['sents'])

        p_bar.update()
        f_data.write(f'=================================  doc{did:03} end  =================================\n\n')
    p_bar.close()
    f_answer.close()
    f_data.close()
    return total_byte_length


def get_pair_documents():
    pair_to_docs = load_json('CTDRED/pair_to_docs.json')
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    mesh_id_to_counts = load_json('CTDRED/mesh_id_to_counts.json')

    fin = open('manual/candidates2.txt', 'r')
    candidate_pairs = []
    for line in fin.readlines():
        line = line.strip()
        if len(line) <= 0:
            continue
        tokens = line.split('\t')
        candidate_pairs.append((tokens[1], tokens[2]))
    fin.close()

    total_byte_length = 0
    err_log = open('manual/err_log.txt', 'w')
    for pid, pair in enumerate(candidate_pairs):
        if pid not in [26]:
            continue
        h, t = pair
        val = pair_to_docs[h + '&' + t]
        val = val[0] + val[1]
        # sample if more than 100 documents
        if len(val) > 100:
            val = random.sample(val, 100)
        documents = get_documents_by_pmids(val, require_all=True)
        total_byte_length += get_single_pair_document(str(pid), documents, pair,
            (mesh_id_to_name[h], mesh_id_to_name[t]), (mesh_id_to_counts[h], mesh_id_to_counts[t]), err_log)
        del documents
    err_log.close()
    print(total_byte_length, (total_byte_length - 500000) / 1000000 * 49)


def get_f1_range():
    import numpy as np
    score_path = 'CTDRED/ctd_cdr_finetune_neg_sample_softmax_range'
    data = load_json('CTDRED/dev_binary_pos.json')
    inst = 0
    for doc in data:
        inst += len(doc['labels'])
    max_f1, max_f1_setting = 0, None
    max_pre, max_pre_setting = 0, None
    max_rec, max_rec_setting = 0, None
    for epoch in range(120):
        titles = load_json(f'{score_path}/dev_binary_pos_pmid2range_{epoch}.json')
        scores = np.load(f'{score_path}/dev_binary_pos_score_{epoch}.npy')
        for threshold in range(-5, 6, 1):
            true_p, pred = 0, 0
            for doc in data:
                labels_set = set()
                for lab in doc['labels']:
                    labels_set.add((lab['h'], lab['t']))
                entities = doc['vertexSet']
                entity_num = len(entities)
                pos = titles[str(doc['pmid'])]
                score_line = scores[pos[0]: pos[1]]
                cur_idx = 0
                for i in range(entity_num):
                    for j in range(entity_num):
                        if not (entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease'):
                            continue
                        score = score_line[cur_idx]
                        if score > threshold:
                            pred += 1
                            if (i, j) in labels_set:
                                true_p += 1
                        cur_idx += 1
            pre, rec = (true_p / pred if pred > 0 else 0), true_p / inst
            f1 = 2 * pre * rec / (pre + rec) if pre + rec > 0 else 0
            if pre > max_pre:
                max_pre = pre
                max_pre_setting = (epoch, threshold, f1, rec)
            if rec > max_rec:
                max_rec = rec
                max_rec_setting = (epoch, threshold, f1, pre)
            if f1 > max_f1:
                max_f1 = f1
                max_f1_setting = (epoch, threshold, pre, rec)
    # ctd_binary_denoise_n15_inter_cdr_ctd
    # 0.5654577909356089 (107, 4, 0.4335981838819523, 0.8125625625625625)
    # ctd_cdr_atloss_p128_n1_l25
    # 0.6052902039270839 (79, 2, 0.5331110052405907, 0.700075075075075)
    # ctd_cdr_contrastive_smloss_range
    # 0.5958264119601329 (113, 5, 0.5091376863023421, 0.7180930930930931)
    print(max_f1, max_f1_setting)
    print(max_pre, max_pre_setting)
    print(max_rec, max_rec_setting)


def reduce_documents_to_two_entities(doc: dict, cid1, cid2):
    if 'labels' in doc:
        del doc['labels']
    new_vertex_set, new_cids = [], []
    assert len(doc['vertexSet']) == len(doc['cids'])
    sents = doc['sents']
    for entity, cid in zip(doc['vertexSet'], doc['cids']):
        if cid in (cid1, cid2):
            new_mentions = []
            for mention in entity:
                sid = mention['sent_id']
                ss, ee = mention['pos']
                if sid >= len(sents) or ss >= len(sents[sid]) or ee > len(sents[sid]):
                    continue
                new_mentions.append(mention)
            assert len(new_mentions) > 0
            new_vertex_set.append(new_mentions)
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


def get_document_set(fid: int):
    fin = open(f'manual/{fid}.txt', encoding='utf-8')
    lines = fin.readlines()
    key_info = lines[1].split('\t')
    cid1, cid2 = key_info[0][1:-1], key_info[1][1:-1]
    pmids = []
    for line in lines:
        if line.startswith('pmid:'):
            pmids.append(int(line.split(' ')[1]))
    fin.close()
    init_db()
    documents = get_documents_by_pmids(pmids, require_all=True)
    for doc in documents:
        reduce_documents_to_two_entities(doc, cid1, cid2)
    save_json(documents, f'manual/{fid}.json')


def gen_part_types_data():
    import copy
    # {1: 0, 2: 2988, 3: 2822}
    for part in ('train_mixed', 'dev', 'test'):
        data = load_json(f'CTDRED/{part}.json')
        data_cd, data_cg, data_dg, data_cdg = [], [], [], []
        pair_to_list = {
            ('Chemical', 'Disease'): data_cd,
            ('Chemical', 'Gene'): data_cg,
            ('Disease', 'Gene'): data_dg,
            ('Chemical', 'Disease', 'Gene'): data_cdg
        }
        for doc in data:
            types = []
            entities = doc['vertexSet']
            for entity in entities:
                types.append(entity[0]['type'])
            cids = doc['cids']
            assert len(entities) == len(types) == len(cids)
            for typ_pair in [('Chemical', 'Disease'), ('Chemical', 'Gene'),
                             ('Disease', 'Gene'), ('Chemical', 'Disease', 'Gene')]:
                all_in = True
                for typ in types:
                    all_in &= typ in typ_pair
                if all_in:
                    doc1 = copy.deepcopy(doc)
                    idx_set = set()
                    new_entities, new_cids, new_rel = [], [], []
                    for idx, typ in enumerate(types):
                        if typ in typ_pair:
                            new_entities.append(entities[idx])
                            new_cids.append(cids[idx])
                            idx_set.add(idx)
                    for rel in doc1['labels']:
                        if rel['h'] in idx_set and rel['t'] in idx_set:
                            new_rel.append({
                                'h': rel['h'], 't': rel['t'], 'r': 'Pos'
                            })
                            new_rel.append({
                                'h': rel['t'], 't': rel['h'], 'r': 'Pos'
                            })
                    doc1['vertexSet'] = new_entities
                    doc1['cids'] = new_cids
                    doc1['labels'] = new_rel
                    pair_to_list[typ_pair].append(doc1)
        print(len(data_cd))
        print(len(data_cg))
        print(len(data_dg))
        print(len(data_cdg))
        pmid_set_cd = set([doc['pmid'] for doc in data_cd])
        pmid_set_cg = set([doc['pmid'] for doc in data_cg])
        pmid_set_dg = set([doc['pmid'] for doc in data_dg])
        assert len(pmid_set_cd & pmid_set_cg) == len(pmid_set_cd & pmid_set_dg) == len(pmid_set_cg & pmid_set_dg) == 0
        save_json(data_cd, f'CTDRED/{part}_cd.json')
        save_json(data_cg, f'CTDRED/{part}_cg.json')
        save_json(data_dg, f'CTDRED/{part}_dg.json')
        save_json(data_cd + data_cg, f'CTDRED/{part}_cd_cg.json')
        save_json(data_cd + data_dg, f'CTDRED/{part}_cd_dg.json')
        save_json(data_cdg, f'CTDRED/{part}_cdg.json')


def temp_func():
    # 24 1640, 25 1056, 42 1560
    # 46 1640, 42 650, 84 1190
    # for part in ('train_mixed', 'dev', 'test'):
    #     data = load_json(f'CTDRED/{part}_cd_cg.json')
    #     print(max(len(doc['labels']) for doc in data),
    #           max(len(doc['vertexSet']) * (len(doc['vertexSet']) - 1) for doc in data))
    # ctd_finetune_cd_cg_combined 52 + dg -> 0 13 544
    # ctd_finetune_cd_dg_combined 44 + cg -> 9 23 2152
    # ctd_finetune_cd_dg_combined 52 + cg -> 18 37 2152

    fin = open('test_paper_new.txt', 'r', encoding='utf-8')
    lines = [line.strip() for line in fin.readlines()]
    cur_title = ''
    results = []
    for line in lines:
        if len(line) > 3 and line[0].isdigit() and (line[1] == '.' or line[2] == '.'):
            if cur_title != '':
                print(cur_title, end=' ')
                assert len(results) == 3
                for idx in range(6):
                    print(f'{round((results[0][idx] + results[1][idx] + results[2][idx]) * 100 / 3, 2):.2f}', end=' & ')
                print()
                results = []
            cur_title = line.split(' ')[1]
        if 'rank46' in line or 'rank23' in line:
            accuracies = re.findall(r'0\.\d+', line)
            assert len(accuracies) == 6
            cur_res = [float(acc) for acc in accuracies]
            cur_res.reverse()
            results.append(cur_res)
    if cur_title != '':
        print(cur_title, end=' ')
        assert len(results) == 3
        for idx in range(6):
            print(f'{round((results[0][idx] + results[1][idx] + results[2][idx]) * 100 / 3, 2):.2f}', end=' & ')
        print()
    fin.close()
    exit()

    cdr_test = load_json('CDR/test_cdr.json')
    candidates = []
    for doc in cdr_test:
        pass
    exit()

    def tt(doc):
        entities = doc['vertexSet']
        entity_num = len(entities)
        cnt = 0
        for i in range(entity_num):
            for j in range(entity_num):
                if entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease':
                    cnt += 1
        return cnt

    data = load_json('CTDRED/ctd_train.json')
    print(sum([len(doc['labels']) for doc in data]))
    print(max([tt(doc) for doc in data]))
    data = load_json('CTDRED/ctd_dev.json')
    print(sum([len(doc['labels']) for doc in data]))
    print(max([tt(doc) for doc in data]))
    data = load_json('CTDRED/ctd_test.json')
    print(max([tt(doc) for doc in data]))
    exit()

    import csv
    fin = open('CTDRED/CTD_chemicals_diseases.csv', 'r', encoding='utf-8')
    reader = csv.reader(fin)
    ctd_pmids = set()
    for line in tqdm(reader):
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
        for pmid in pmids.split('|'):
            if pmid != '':
                ctd_pmids.add(int(pmid))
    fin.close()
    overall_set = get_documents_by_pmids(list(ctd_pmids), require_all=False)
    overall_set = [doc for doc in overall_set if doc != {}]
    train_set, dev_set, test_set = [], [], []
    for doc in overall_set:
        if len(doc['labels']) == 0:
            continue
        rand = random.random()
        if rand <= 0.6:
            train_set.append(doc)
        elif rand <= 0.8:
            dev_set.append(doc)
        else:
            test_set.append(doc)
    print(len(overall_set), len(train_set), len(dev_set), len(test_set))
    save_json(train_set, 'CTDRED/ctd_train.json')
    save_json(dev_set, 'CTDRED/ctd_dev.json')
    save_json(test_set, 'CTDRED/ctd_test.json')
    exit()

    import numpy as np
    true_p, predict, instance = 0, 0, 0
    with open('manual/0_seg_ans.txt', 'r') as fin:
        lines = [line.strip() for line in fin.readlines() if len(line.strip()) > 0]
    data = load_json('manual/0_pmc_segments_sample.json')
    assert len(lines) == len(data)
    scores = np.load('manual/test/ctd_cdr_finetune_neg_sample_softmax/0_pmc_segments_sample_score_10.npy')
    titles = load_json('manual/test/ctd_cdr_finetune_neg_sample_softmax/0_pmc_segments_sample_pmid2range_10.json')
    idx = titles['29096648_PMC5669003_251_258']
    print(scores[idx[0]:idx[1]])
    for lid, line in enumerate(lines):
        pmid, label, _ = line.split(',')
        if label == '1':
            instance += 1
        idx = titles[pmid]
        assert idx[1] - idx[0] == 1
        score = scores[idx[0]]
        if score >= 0.5:
            predict += 1
            if label == '1':
                true_p += 1
            else:
                print('false positive:', lid, pmid, score)
        elif label == '1':
            print('false negative:', lid, pmid, score)
    print(true_p, predict, instance)
    exit()

    doc = load_json('manual/test.json')
    passages = doc['passages']
    title = ''
    texts = []
    section_types = []
    annotations = []
    accu_offset = 0
    accu_text = ''
    for section in passages:
        raw_text = section['text']
        strip_text = raw_text.strip()
        if len(strip_text) == 0:
            continue
        if title == '':
            title = strip_text
        l_pos = raw_text.find(strip_text[0])
        assert l_pos != -1
        sent_offset = section['offset'] + l_pos

        section_types.append(section['infons']['section_type'])
        texts.append(strip_text)
        new_anno = []
        anno = section['annotations']
        for entity in anno:
            new_loc = []
            for loc in entity['locations']:
                if loc['offset'] < section['offset']:
                    continue
                loc['offset'] = loc['offset'] - sent_offset + accu_offset
                new_loc.append(loc)
            if len(new_loc) > 0:
                entity['locations'] = new_loc
                new_anno.append(entity)

        accu_offset += len(strip_text) + 1
        annotations += new_anno
        accu_text += strip_text + ' '

    text = ' '.join(texts)
    for entity in annotations:
        for loc in entity['locations']:
            off, le = loc['offset'], loc['length']
            if text[off:(off+le)] == entity['text']:
                continue
            print(text[off:(off+le)], entity['text'])

    pmid, pmcid = doc['_id'].split('|')
    result = {
        'pmid': pmid,
        'pmcid': pmcid,
        'title': title,
        'section_types': section_types,
        'texts': texts,
        'entities': annotations
    }
    doc = pubtator_to_docred(result, [])
    return doc


def get_main_articles(pair, seg_idx):
    doc2labels = {key: tuple(value) for key, value in load_json('CTDRED/ctd_doc_to_labels.json').items()}
    # pair, seg_idx = 'D014640&D008586', 0
    # pair, seg_idx = 'D013752&D004403', 25
    # pair, seg_idx = 'D005480&D010003', 1
    h_cid, t_cid = pair.split('&')
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    h_name, t_name = mesh_id_to_name[h_cid], mesh_id_to_name[t_cid]
    documents = search_get_pubmed(entities=[h_name, t_name], ent_pair=(h_cid, t_cid),
                                  ret_max=1000, require_contain=False, pmc=True)[0]
    save_json(documents, f'manual/{seg_idx}_pmc.json')
    processed = []
    print(len(documents))
    for did, val in tqdm(documents.items()):
        red_doc = pubtator_to_docred(val, doc2labels[did] if did in doc2labels else [])
        cids = red_doc['cids']
        assert h_cid in cids and t_cid in cids
        processed.append(red_doc)
    print(len(processed))

    for doc in processed:
        sents = doc['sents']
        new_sents = []
        sid_map = {}
        entities = doc['vertexSet']
        for sid, sent in enumerate(sents):
            is_null = True
            for token in sent:
                if token.strip() != "":
                    is_null = False
                    break
            if not is_null:
                sid_map[sid] = len(new_sents)
                new_sents.append(sent)
        for entity in entities:
            for mention in entity:
                assert mention['sent_id'] in sid_map
                mention['sent_id'] = sid_map[mention['sent_id']]
        doc['sents'] = new_sents

    for doc in processed:
        sents = doc['sents']
        entities = doc['vertexSet']
        sent_to_off = []
        new_sents = []
        for sid, sent in enumerate(sents):
            null_s, null_e, s_len = 0, 0, len(sent)
            while null_s < s_len and sent[null_s] == " ":
                null_s += 1
            while null_e < s_len and sent[s_len-1-null_e] == " ":
                null_e += 1
            try:
                assert null_s + null_e < s_len
            except AssertionError as err:
                print('------------------')
                print(doc['pmcid'])
                print(sent, null_s, null_e)
                raise err
            sent_to_off.append((null_s, null_e, s_len))
            new_sents.append(sent[null_s:(s_len-null_e)])
        doc['sents'] = new_sents
        for entity in entities:
            for mention in entity:
                sid = mention['sent_id']
                null_s, null_e, s_len = sent_to_off[sid]
                if null_s > 0 or null_e > 0:
                    start, end = mention['pos']
                    try:
                        assert start >= null_s and end <= s_len - null_e
                    except AssertionError as err:
                        print(sents[sid])
                        print(null_s, null_e)
                        print(mention)
                        raise err
                    mention['pos'] = [start - null_s, end - null_s]
    save_json(processed, f'manual/{seg_idx}_pmc_processed.json')


def get_article_segments(pair, seg_idx):
    segment_max_len = 15 - 4
    # pair, seg_idx = 'D014640&D008586', 0
    # pair, seg_idx = 'D013752&D004403', 25
    # pair, seg_idx = 'D005480&D010003', 1
    h_cid, t_cid = pair.split('&')
    data = load_json(f'manual/{seg_idx}_pmc_processed.json')
    print(len(data))
    segments = []
    for doc in data:
        cids = doc['cids']
        entities = doc['vertexSet']
        cid_to_id = {cid: i for i, cid in enumerate(cids)}
        hid, tid = cid_to_id[h_cid], cid_to_id[t_cid]
        h_sids, t_sids = [], []
        for mention in entities[hid]:
            h_sids.append(mention['sent_id'])
        for mention in entities[tid]:
            t_sids.append(mention['sent_id'])
        h_sids.sort()
        t_sids.sort()
        sid_bound = -1
        sent_num = len(doc['sents'])
        for hs in h_sids:
            if hs <= sid_bound:
                continue
            for ts in t_sids:
                if ts <= sid_bound:
                    continue
                if abs(hs - ts) < segment_max_len:
                    chosen = sorted((hs, ts))
                    chosen = max(0, chosen[0] - 2), min(sent_num, chosen[1] + 3)
                    sid_bound = chosen[1] - 1

                    # process segment
                    pmsid = f'{doc["pmid"]}_{doc["pmcid"]}_{chosen[0]}_{chosen[1]}'
                    new_sents = doc['sents'][chosen[0]:chosen[1]]
                    new_cids = []
                    new_vertex_set = []
                    assert len(doc['cids']) == len(doc['vertexSet'])
                    for cid, entity in zip(doc['cids'], doc['vertexSet']):
                        mentions = []
                        for mention in entity:
                            if chosen[0] <= mention['sent_id'] < chosen[1]:
                                mention['sent_id'] -= chosen[0]
                                mentions.append(mention)
                        if len(mentions) > 0:
                            new_vertex_set.append(mentions)
                            new_cids.append(cid)
                    segments.append({
                        'pmsid': pmsid,
                        'sents': new_sents,
                        'cids': new_cids,
                        'vertexSet': new_vertex_set,
                    })
                    if hs <= sid_bound:
                        break
    print(len(segments))
    print(fix_ner_by_search(segments))
    save_json(segments, f'manual/{seg_idx}_pmc_segments.json')


def gen_segment_annotation_files(pair, seg_idx):
    err_log = open('manual/err_log_seg.txt', 'w')
    random.seed(100)
    # pair, seg_idx = 'D014640&D008586', 0
    # pair, seg_idx = 'D013752&D004403', 25
    # pair, seg_idx = 'D005480&D010003', 1
    documents = load_json(f'manual/{seg_idx}_pmc_segments.json')
    if len(documents) > 100:
        documents = random.sample(documents, 100)
    h_cid, t_cid = pair.split('&')
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    h_name, t_name = mesh_id_to_name[h_cid], mesh_id_to_name[t_cid]
    mesh_id_to_counts = load_json('CTDRED/mesh_id_to_counts.json')
    h_count, t_count = mesh_id_to_counts[h_cid], mesh_id_to_counts[t_cid]
    total_byte_length = get_single_pair_document(f'{seg_idx}_seg', documents, (h_cid, t_cid),
                                                 (h_name, t_name), (h_count, t_count), err_log, pmc=True)
    print(total_byte_length)
    for doc in documents:
        reduce_documents_to_two_entities(doc, h_cid, t_cid)
    save_json(documents, f'manual/{seg_idx}_pmc_segments_sample.json')
    err_log.close()


def combine_second_round():
    first_round = 'manual/manual_ans1'
    second_round_ori = 'manual/manual_ans2_ori'
    second_round = 'manual/manual_ans2'
    final_round = 'manual/manual_new'
    first_round_f = sorted(os.listdir(first_round))
    second_round_ori_f = sorted(os.listdir(second_round_ori))
    second_round_f = sorted(os.listdir(second_round))
    final_round_f = sorted(os.listdir(final_round))
    print(len(first_round_f), len(second_round_f), len(second_round_ori_f), len(final_round_f))
    assert first_round_f == second_round_f == second_round_ori_f
    for final in final_round_f:
        assert final in final_round_f
    for first_n, second_n, second_ori_n in zip(first_round_f, second_round_f, second_round_ori_f):
        with open(os.path.join(first_round, first_n)) as fin:
            first_l = [line.strip().split(',') for line in fin.readlines() if len(line.strip()) > 0]
            assert all(len(item) == 3 for item in first_l)
        with open(os.path.join(second_round, second_n)) as fin:
            second_l = [line.strip().split(',') for line in fin.readlines() if len(line.strip()) > 0]
            assert all(len(item) == 3 for item in second_l)
        with open(os.path.join(second_round_ori, second_ori_n)) as fin:
            second_ori_l = [line.strip().split(',') for line in fin.readlines() if len(line.strip()) > 0]
            for item in second_l:
                if len(item) != 3:
                    print(second_n, item)
            assert all(len(item) == 3 for item in second_ori_l)
        assert len(first_l) == len(second_l) == len(second_ori_l)
        if first_n in final_round_f:
            # 已经合并好
            with open(os.path.join(final_round, first_n)) as fin:
                final_l = [line.strip().split(',') for line in fin.readlines() if len(line.strip()) > 0]
                assert all(len(item) == 3 and float(item[2]) in [1, 2, 3, 1.5, 2.5] for item in final_l)
                assert len(final_l) == len(first_l)
            for first, second, second_ori, final in zip(first_l, second_l, second_ori_l, final_l):
                # 如果第二轮结果有修改
                assert first[0] == second[0] == second_ori[0] == final[0]
            continue
        fout = open(os.path.join(final_round, first_n), 'w')
        for first, second, second_ori in zip(first_l, second_l, second_ori_l):
            assert first[0] == second[0] == second_ori[0]
            first_tag, first_rank = int(first[1]), int(first[2])
            second_tag, second_rank = int(second[1]), int(second[2])
            second_ori_tag, second_ori_rank = int(second_ori[1]), int(second_ori[2])
            # 如果第二轮结果有修改，取改正结果
            if (second_tag, second_rank) != (second_ori_tag, second_ori_rank):
                fout.write(f'{first[0]},{second_tag},{second_rank}\n')
            # 如果一二轮标签不一致，取二轮结果
            elif first_tag != second_tag:
                fout.write(f'{first[0]},{second_tag},{second_rank}\n')
            # 取平均
            else:
                avg_rank = round((first_rank + second_rank) / 2, 1)
                assert avg_rank in [1, 2, 3, 1.5, 2.5]
                fout.write(f'{first[0]},{second_tag},{avg_rank}\n')
        fout.close()


if __name__ == '__main__':
    # combine_second_round()
    # get_main_articles()
    # get_article_segments()
    # gen_segment_annotation_files()
    temp_func()
    exit()
    # get_gene_descriptions()
    # exit()
    # temp_func()
    # gen_part_types_data()
    # get_f1_range()
    # gen_pair_to_docs()
    # get_sample_pairs()
    # get_pair_documents()
    # for fid in range(50):
    #     get_document_set(fid)
