import os
import torch
from transformers import AutoTokenizer


TOKENIZER = AutoTokenizer.from_pretrained('../huggingface/scibert_scivocab_cased' if os.path.exists(
    '../huggingface/scibert_scivocab_cased') else 'allenai/scibert_scivocab_cased')
CONFIG = {
    'token_padding': 1024,
    'mention_padding': 3,
    'entity_padding': 2,
    'crop_documents': False,
    'crop_mention_option': 4,
    'entity_marker_type': 't',
}


def process_denoise(data, mode: str):
    global CONFIG
    assert mode == 'test'

    documents, attn_masks, word_positions, head_ids, tail_ids, pos_ids, titles = [], [], [], [], [], [], []
    pmid_key = 'pmid' if len(data) > 0 and 'pmid' in data[0] else 'pmsid'
    for doc in data:
        document, attn_mask, position = process_document(doc, mode)
        documents.append(document)
        attn_masks.append(attn_mask)
        word_positions.append(position)

        entity_num = len(doc['vertexSet'])
        entities = doc['vertexSet']
        pairs = [(i, j) for i in range(entity_num) for j in range(entity_num) if
                    entities[i][0]['type'] == 'Chemical' and entities[j][0]['type'] == 'Disease']
        head_ids.append([pair[0] for pair in pairs])
        tail_ids.append([pair[1] for pair in pairs])
        titles.append(str(doc[pmid_key]))

    # dynamic pad positions
    entity_padding = max(len(item) for item in word_positions)
    for item in word_positions:
        for _ in range(entity_padding - len(item)):
            item.append([0] * CONFIG['mention_padding'])
    # dynamic pad head/tail ids
    test_sample_limit = max(len(item) for item in head_ids)
    sample_counts = []
    for head_id, tail_id in zip(head_ids, tail_ids):
        assert len(head_id) == len(tail_id)
        sample_counts.append(len(head_id))
        head_id += [0] * (test_sample_limit - len(head_id))
        tail_id += [0] * (test_sample_limit - len(tail_id))

    return {
        'documents': torch.LongTensor(documents),
        'attn_mask': torch.FloatTensor(attn_masks),
        'word_pos': torch.LongTensor(word_positions),
        'head_ids': torch.LongTensor(head_ids),
        'tail_ids': torch.LongTensor(tail_ids),
        'titles': titles,
        'sample_counts': sample_counts
    }


def process_document(data, mode):
    global CONFIG, TOKENIZER
    assert mode == 'test'
    if CONFIG['crop_documents']:
        document_crop(data)
    sentence_mention_crop(data, mode, CONFIG['crop_mention_option'])
    sentences = [[TOKENIZER.tokenize(word) for word in sent] for sent in data['sents']]

    entities = data['vertexSet']
    for i, mentions in enumerate(entities):
        for mention in mentions:
            if CONFIG['entity_marker_type'] not in ['t', 't-m']:
                tmp: list = sentences[mention['sent_id']][mention['pos'][0]]
                if CONFIG['entity_marker_type'] == 'mt':
                    # both mention and type
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [f'[unused{(i << 1) + 1}]', mention['type'], '*'] + tmp
                else:
                    # Config.entity_marker_type == 'm', only mention
                    sentences[mention['sent_id']][mention['pos'][0]] = [f'[unused{(i << 1) + 1}]'] + tmp
            else:
                # Config.entity_marker_type in ['t', 't-m'], blank all mention, only type; t-m: type, *, [MASK]
                for pos in range(mention['pos'][0], mention['pos'][1]):
                    sentences[mention['sent_id']][pos] = []
                if CONFIG['entity_marker_type'] == 't':
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [f'[unused{(i << 1) + 1}]', mention['type'], '*', '[unused0]']
                else:
                    assert CONFIG['entity_marker_type'] == 't-m'
                    sentences[mention['sent_id']][mention['pos'][0]] = \
                        [mention['type'], '*', '[unused0]']
            if CONFIG['entity_marker_type'] != 't-m':
                sentences[mention['sent_id']][mention['pos'][1] - 1].append(f'[unused{(i + 1) << 1}]')

    word_position, document = [], ['[CLS]']
    for sent in sentences:
        word_position.append([])
        for word in sent:
            word_position[-1].append(len(document))
            document += word
    word_position.append([len(document)])

    # pad each document
    if len(document) < CONFIG['token_padding']:
        document.append('[SEP]')
        document += ['[PAD]'] * (CONFIG['token_padding'] - len(document))
        attn_mask = [1] * len(document) + [0] * (CONFIG['token_padding'] - len(document))
    else:
        document = document[:(CONFIG['token_padding'] - 1)] + ['[SEP]']
        attn_mask = [1] * CONFIG['token_padding']

    positions = []
    for entity in entities:
        cur_entity = []
        for mention in entity:
            if word_position[mention['sent_id']][mention['pos'][0]] < CONFIG['token_padding']:
                cur_entity.append(word_position[mention['sent_id']][mention['pos'][0]])
            if len(cur_entity) == CONFIG['mention_padding']:
                break
        positions.append(cur_entity)
    # padding length of mention number to 3
    for i in range(len(positions)):
        if len(positions[i]) == 0:
            positions[i] = [0] * CONFIG['mention_padding']
        positions[i] += [positions[i][0]] * (CONFIG['mention_padding'] - len(positions[i]))

    return TOKENIZER.convert_tokens_to_ids(document), attn_mask, positions


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
        # 会影响 labels 中的其他实体对，怎么办？
        # 一种解决方法是两两推断，但那样完全放弃了 multi-hop，而且测试的时候效率较低
        # 训练的时候多 option3 mask，测试时两个两个过？
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
