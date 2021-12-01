import json
from search_utils import adaptive_load
from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, chunked


db = SqliteDatabase('CTDRED/documents.db')


class BaseModel(Model):
    class Meta:
        database = db


class Documents(BaseModel):
    id = AutoField()
    pmid = IntegerField(null=False, index=True, unique=True, column_name='pmid')
    content = TextField(null=False, column_name='content')


def init_db():
    db.connect()
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


if __name__ == '__main__':
    init_db()
    dump_data()
