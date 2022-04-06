import json
from .search_utils import adaptive_load, load_json
from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase


db = SqliteDatabase('CTDRED/documents.db')


class BaseModel(Model):
    class Meta:
        database = db


class Documents(BaseModel):
    id = AutoField()
    pmid = IntegerField(null=False, index=True, unique=True, column_name='pmid')
    content = TextField(null=False, column_name='content')


def init_db():
    db.connection()
    db.create_tables([Documents])


def dump_data():
    for part in ['train_mixed', 'dev', 'test', 'negative_train_mixed', 'negative_dev', 'negative_test',
                 'negative_train_extra', 'negative_dev_extra', 'negative_test_extra', 'ctd_extra', 'extra_batch']:
        data = adaptive_load(f'CTDRED/{part}_binary_pos')
        cur_batch, cur_cnt = [], 0
        for doc in data:
            cur_batch.append((doc['pmid'], json.dumps(doc)))
            if len(cur_batch) == 100:
                with db.atomic():
                    Documents.insert_many(cur_batch, fields=[Documents.pmid, Documents.content]).execute()
                cur_cnt += len(cur_batch)
                cur_batch = []
                print(f'{part} processed: {cur_cnt:07}', end='\r')
        if len(cur_batch) > 0:
            with db.atomic():
                Documents.insert_many(cur_batch, fields=[Documents.pmid, Documents.content]).execute()
            cur_cnt += len(cur_batch)
        print(f'{part} processed: {cur_cnt:07}')
        del data


def get_documents_by_pmids(pmids: list, require_all: bool):
    """找不到的 pmid 返回空字典 {}"""
    db_ret = Documents.select(Documents.pmid, Documents.content).where(Documents.pmid.in_(pmids))
    if require_all and len(pmids) != len(db_ret):
        raise RuntimeError('require all error!')
    ret = [{} for _ in range(len(pmids))]
    pmid2idx = {pmid: i for i, pmid in enumerate(pmids)}
    for doc in db_ret:
        ret[pmid2idx[doc.pmid]] = json.loads(doc.content)
    return ret


def fix_pubtator_ner():
    from tqdm import tqdm
    from typing import List, Dict
    """硬匹配修复 pubtator NER 不全的问题"""
    init_db()
    en_punc = '.,<>?/\\[]{};:\'\"|=+-_()*&^%$#@!~` '
    mesh_id_to_name = load_json('CTDRED/mesh_id_to_name.json')
    all_iter = Documents.select(Documents.id, Documents.pmid, Documents.content)
    update_batch, update_cnt, all_cnt = [], 0, 0
    p_bar = tqdm()
    for doc in all_iter:
        p_bar.update()
        all_cnt += 1
        data = json.loads(doc.content)
        data_dirty = False
        assert doc.pmid == data['pmid']
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
                men_name = [token.strip(en_punc).lower() for token in men_name]
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
                            if sent_tokens[start:(start+name_len)] == candidate_name:
                                # add to vertexSet
                                entity.append({
                                    'name': ' '.join(sent[start:(start+name_len)]),
                                    'sent_id': sid,
                                    'pos': [start, start + name_len],
                                    'type': ent_type
                                })
                                data_dirty = True
                                break
        if data_dirty:
            doc.content = json.dumps(data)
            update_batch.append(doc)
            update_cnt += 1
            if len(update_batch) == 100:
                with db.atomic():
                    Documents.bulk_update(update_batch, fields=[Documents.content])
                del update_batch
                update_batch = []
    p_bar.close()
    if len(update_batch) > 0:
        with db.atomic():
            Documents.bulk_update(update_batch, fields=[Documents.content])
    print('updated documents:', update_cnt, 'all documents:', all_cnt)


if __name__ == '__main__':
    init_db()
    dump_data()
    fix_pubtator_ner()
