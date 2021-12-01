import random
from tqdm import tqdm
from ncbi_api import baidu_translate
from search_db import init_db, get_documents_by_pmids
from search_utils import save_json, load_json, adaptive_load


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
    candidate_pairs, candidate_pairs_nctd = [], []

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

    def batch_translate_write(cur_mark_sents, cur_raw_sents, fout_data, fout_log):
        # 批量翻译
        zh_translation, load_length = None, 0
        for _ in range(10):
            try:
                # zh_translation = ['' for _ in range(len(cur_raw_sents))]
                zh_translation, load_length = baidu_translate(cur_raw_sents)
                break
            except Exception as err:
                print(err)
        if zh_translation is None:
            raise RuntimeError('baidu translation failure!')
        if len(zh_translation) != len(cur_raw_sents):
            to_translate = '\n'.join(cur_raw_sents)
            fout_log.write(f'{doc["pmid"]}\n')
            fout_log.write(f'{to_translate}\n\n')
            zh_translation = ['' for _ in range(len(cur_raw_sents))]
        assert len(zh_translation) == len(cur_raw_sents) == len(cur_mark_sents)
        for ssid in range(len(cur_raw_sents)):
            ss, mm = cur_mark_sents[ssid]
            zh_sent = zh_translation[ssid]
            fout_data.write(f'{"*** " if mm else ""}SENTENCE {ssid}: {ss}\n')
            fout_data.write(f'{"*** " if mm else ""}句子 {ssid}: {zh_sent}\n\n')
        return load_length, len(cur_raw_sents)

    # 选实体对和文章
    filter_list = ['D002241', 'D005947', 'D002244', 'D007501', 'D002248', 'D011041', 'D003300', 'D005483',
                   'D000073893', 'D014867', 'D010100', 'D010758', 'D009369', 'D003643', 'D003681', 'D006261',
                   'D009325', 'D063646']
    head_entities, head_entities_nctd = {}, {}
    for pair, val in pair_to_docs.items():
        cur_doc_num = len(val[0] + val[1])
        h, t = pair.split('&')
        h_occur, t_occur = mesh_id_to_counts[h], mesh_id_to_counts[t]
        if upper_bound > cur_doc_num >= lower_bound and upper_ent_bound > max(h_occur, t_occur) >= lower_ent_bound:
            if h not in mesh_id_to_name or t not in mesh_id_to_name or h not in ctd_entities or t not in ctd_entities:
                continue
            if h in filter_list or t in filter_list:
                continue
            t_name = mesh_id_to_name[t].lower()
            if 'disease' in t_name or 'pain' in t_name or 'disorder' in t_name or 'failure' in t_name \
                    or 'disorders' in t_name or 'loss' in t_name or 'gain' in t_name or 'insufficiency' in t_name\
                    or 'heart' in t_name:
                continue
            cnt_thred += 1
            if lab_upper * cur_doc_num > len(val[1]) >= lab_lower * cur_doc_num:
                cnt_lab_thred += 1
                doc_cnt += cur_doc_num
                if pair in ctd_relations:
                    head_entities.setdefault(h, [])
                    head_entities[h].append(t)
                else:
                    head_entities_nctd.setdefault(h, [])
                    head_entities_nctd[h].append(t)
    for h in random.sample(head_entities.keys(), 25):
        candidate_pairs.append(h + '&' + random.choice(head_entities[h]))
    for h in random.sample(head_entities_nctd.keys(), 25):
        candidate_pairs.append(h + '&' + random.choice(head_entities_nctd[h]))

    fout = open('manual/candidates2.txt', 'w')
    # half in ctd and half not in ctd
    # candidate_pairs = random.sample(candidate_pairs, 25) + random.sample(candidate_pairs_nctd, 25)

    # replace common entities
    def part_replace(replace_list: list):
        for idx in replace_list:
            if idx < 25:
                candidate_pairs[idx] = random.choice(candidate_pairs)
            else:
                candidate_pairs[idx] = random.choice(candidate_pairs_nctd)

    # part_replace([])

    total_byte_length = 0
    err_log = open('manual/err_log.txt', 'w')
    for pid, pair in enumerate(candidate_pairs):
        # if pid in [0, 25]:
        #     continue
        f_data = open(f'manual/{pid}.txt', 'w', encoding='utf-8')
        f_answer = open(f'manual/{pid}_ans.txt', 'w')
        val = pair_to_docs[pair]
        val = val[0] + val[1]
        # sample if more than 100 documents
        if len(val) > 100:
            val = random.sample(val, 100)
        h, t = pair.split('&')
        fout.write(f'{pid}\t{h}\t{t}\t{len(val)}\t{mesh_id_to_name[h]}\t{mesh_id_to_counts[h]}'
                   f'\t{mesh_id_to_name[t]}\t{mesh_id_to_counts[t]}\n')
        # continue
        f_data.write('[head_MESH]\t[tail_MESH]\t[num_of_docs]\t[head_name]\thead_occurrence\t[tail_name]'
                     '\ttail_occurrence\n')
        f_data.write(f'[{h}]\t[{t}]\t[{len(val)}]\t[{mesh_id_to_name[h]}]\t{mesh_id_to_counts[h]}'
                     f'\t[{mesh_id_to_name[t]}]\t{mesh_id_to_counts[t]}\n\n')
        # get documents
        documents = get_documents_by_pmids(val, require_all=True)
        p_bar = tqdm(len(documents), desc=str(pid))
        for did, doc in enumerate(documents):
            f_answer.write(f'{doc["pmid"]},\n')
            f_data.write(f'================================= doc{did:03} begin =================================\n\n')
            f_data.write('[head_MESH]\t[tail_MESH]\t[num_of_docs]\t[head_name]\t[tail_name]\n')
            f_data.write(f'[{h}]\t[{t}]\t[{len(val)}]\t[{mesh_id_to_name[h]}]\t[{mesh_id_to_name[t]}]\n\n')
            f_data.write(f'pmid: {doc["pmid"]} sentences:\n\n')
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
                    trans_len, written_num = batch_translate_write(cc_mark_sents, cc_raw_sents, f_data, err_log)
                    total_byte_length += trans_len
                    written_sent_num += written_num
                    cc_mark_sents, cc_raw_sents, cc_raw_sents_len = [], [], 0

                cc_mark_sents.append((text, is_star_sent))
                cc_raw_sents.append(raw_sent)
                cc_raw_sents_len += len(raw_sent) + 1

            trans_len, written_num = batch_translate_write(cc_mark_sents, cc_raw_sents, f_data, err_log)
            total_byte_length += trans_len
            written_sent_num += written_num
            assert written_sent_num == len(doc['sents'])

            p_bar.update()
            f_data.write(f'=================================  doc{did:03} end  =================================\n\n')
        del documents
        p_bar.close()
        f_answer.close()
        f_data.close()
    err_log.close()
    fout.close()
    print(len(pair_to_docs), cnt_thred, cnt_lab_thred, doc_cnt)
    print(total_byte_length, (total_byte_length - 500000) / 1000000 * 49)


if __name__ == '__main__':
    # gen_pair_to_docs()
    get_sample_pairs()
