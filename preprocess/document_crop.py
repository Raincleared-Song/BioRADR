def document_crop(doc):
    """
    缩减文章规模，规则如下：
    如果句子不包含实体，且上下句中任何一句不存在或不包含实体，则删去此句
    """
    entities = doc['vertexSet']
    sents = doc['sents']
    sent_entity_appear = set()
    for entity in entities:
        for mention in entity:
            sent_entity_appear.add(mention['sent_id'])
    new_sents = []
    sid_map = {}
    for sid, sent in enumerate(sents):
        if sid not in sent_entity_appear and \
                ((sid - 1) not in sent_entity_appear or (sid + 1) not in sent_entity_appear):
            continue
        new_sents.append(sent)
        sid_map[sid] = len(sid_map)
    for entity in entities:
        for mention in entity:
            mention['sent_id'] = sid_map[mention['sent_id']]
    doc['sents'] = new_sents


def test_cp_negative(item: dict):
    num_c, num_g = 0, 0
    for entity in item['vertexSet']:
        if entity[0]['type'].lower().startswith('chemical'):
            num_c += 1
        else:
            num_g += 1
    return num_c * num_g > len([lab for lab in item['labels'] if lab['r'] != 'NA'])


def sentence_mention_crop(doc, mode: str, option: int):
    """
    缩减可能造成干扰的 mention。
    对于一个在 labels 中的实体对，它们共同出现在了某些句子中(共通句)，那么：
    option == 0: 什么都不做；
    option == 1: 只处理一个共通句的情况，隐藏其他句子里的 mention
    option == 2: 处理所有含共通句的情况，隐藏其他句子里的 mention
    option == 3: 根据所有 label 确定保留的句子，适用于多实体的情况，公共句为 1 句时删除其他，否则保留全部
    option == 4: 有公共句时就删除其他，否则保留全部
    """
    if option == 0:
        return
    entities = doc['vertexSet']
    entity_num = len(entities)
    # for test mode
    if mode == 'test':
        doc_labels = []
        for i in range(entity_num):
            for j in range(entity_num):
                if entities[i][0]['type'].lower() == 'chemical' and entities[j][0]['type'].lower() == 'disease':
                    doc_labels.append({'h': i, 't': j})
    else:
        doc_labels = doc['labels']
    assert len(doc_labels) > 0
    # sent_id reserved for each entity
    reserve_mentions = None
    for lab in doc_labels:
        h, t = lab['h'], lab['t']
        assert entities[h][0]['type'] == 'Chemical' and entities[t][0]['type'] == 'Disease'
        pos_h = set([mention['sent_id'] for mention in entities[h]])
        pos_t = set([mention['sent_id'] for mention in entities[t]])
        cur_intersect = pos_h & pos_t
        in_len = len(cur_intersect)

        if (option == 1 and in_len == 1 or option == 2 and in_len > 0) and (len(pos_h) > in_len or len(pos_t) > in_len):
            # mask all the other mentions
            new_mention_h, new_mention_t = [], []
            for mention in entities[h]:
                if mention['sent_id'] in cur_intersect:
                    new_mention_h.append(mention)
            for mention in entities[t]:
                if mention['sent_id'] in cur_intersect:
                    new_mention_t.append(mention)
            assert len(set([mention['sent_id'] for mention in new_mention_h])) == in_len
            assert len(set([mention['sent_id'] for mention in new_mention_t])) == in_len
            doc['vertexSet'][h] = new_mention_h
            doc['vertexSet'][t] = new_mention_t
        elif option >= 3 and in_len > 0:
            if reserve_mentions is None:
                reserve_mentions = [set() for _ in range(entity_num)]
            if option == 3 and in_len == 1 or option == 4:
                reserve_mentions[h] |= cur_intersect
                reserve_mentions[t] |= cur_intersect
    if option >= 3 and reserve_mentions is not None:
        new_vertex = []
        for entity, sid_set in zip(entities, reserve_mentions):
            cur_entity = []
            for mention in entity:
                if len(sid_set) == 0 or mention['sent_id'] in sid_set:
                    cur_entity.append(mention)
            new_vertex.append(cur_entity)
        doc['vertexSet'] = new_vertex
