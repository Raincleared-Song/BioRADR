import os
import json
import jsonlines
from tqdm import tqdm
from typing import List, Dict


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


def adaptive_load(path: str):
    """
    dynamic load json or jsonl files
    :param path: prefix of the file
    :return: an iterator
    """
    if os.path.exists(path + '.json'):
        return iter(load_json(path + '.json'))
    else:
        return iter(jsonlines.open(path + '.jsonl', mode='r'))


def print_json(obj, file=None):
    print(json.dumps(obj, indent=4, separators=(', ', ': '), ensure_ascii=False), file=file)


def fix_ner_by_search(documents: list):
    en_punc = '.,<>?/\\[]{};:\'\"|=+-_()*&^%$#@!~` '
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    p_bar = tqdm(range(len(documents)))
    all_cnt, update_cnt = 0, 0
    for did, data in enumerate(documents):
        p_bar.update()
        all_cnt += 1
        data_dirty = False
        entities = data['vertexSet']
        cids = data['cids']
        sents = data['sents']
        entity_num = len(entities)
        assert len(cids) == entity_num
        cur_sid_to_posed: Dict[int, List[int]] = {sid: [] for sid in range(len(sents))}
        for entity in entities:
            for mention in entity:
                positions = cur_sid_to_posed[mention['sent_id']]
                for pos in range(mention['pos'][0], mention['pos'][1]):
                    positions.append(pos)
        for eid, entity in enumerate(entities):
            # token length -> [tokens]
            ent_type = entity[0]['type']
            filter_names: Dict[int, List[List[str]]] = {}
            if cids[eid] in mesh_id_to_name:
                std_name = mesh_id_to_name[cids[eid]].split(' ')
                std_name = [token.strip(en_punc).lower() for token in std_name]
                filter_names.setdefault(len(std_name), [])
                filter_names[len(std_name)].append(std_name)
            for mention in entity:
                men_name = mention['name']
                for ch in en_punc:
                    men_name = men_name.replace(ch, ' ')
                men_name = men_name.split(' ')
                men_name = [token.lower() for token in men_name]
                filter_names.setdefault(len(men_name), [])
                filter_names[len(men_name)].append(men_name)
            if 1 in filter_names:
                # remove those single word shorter than 3 characters
                new_single_names = [name for name in filter_names[1] if len(name[0]) >= 3]
                if len(new_single_names) > 0:
                    filter_names[1] = new_single_names
                else:
                    del filter_names[1]
            # complete NER results
            for name_len, name_tokens in filter_names.items():
                for sid, sent in enumerate(sents):
                    if len(sent) < name_len:
                        continue
                    sent_tokens = [token.strip(en_punc).lower() for token in sent]
                    sent_positions = cur_sid_to_posed[sid]
                    for start in range(0, len(sent) - name_len + 1):
                        pos_passed = True
                        for tid in range(start, start + name_len):
                            if tid in sent_positions:
                                pos_passed = False
                                break
                        if not pos_passed:
                            continue
                        for candidate_name in name_tokens:
                            if sent_tokens[start:(start + name_len)] == candidate_name:
                                # add to vertexSet
                                entity.append({
                                    'name': ' '.join(sent[start:(start + name_len)]),
                                    'sent_id': sid,
                                    'pos': [start, start + name_len],
                                    'type': ent_type
                                })
                                data_dirty = True
                                break
        if data_dirty:
            update_cnt += 1
    p_bar.close()
    return all_cnt, update_cnt
