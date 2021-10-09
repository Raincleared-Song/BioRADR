import json
import copy
import random


def load_json(path: str):
    print(f'loading file {path} ......')
    file = open(path)
    res = json.load(file)
    file.close()
    return res


def print_json(obj):
    print(json.dumps(obj, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))


if __name__ == '__main__':
    use_cp = True
    if use_cp:
        data = load_json('Chemprot/chemprot_dev_sent_fr.json')
        rel_to_id = load_json('Chemprot/chemprot_relation_to_id.json')
        result_nod = load_json('checkpoint/test_cps_finetune_new_lr03/result.json')
        result_wod = load_json('checkpoint/test_cps_finetune_new_lr03_d5/result.json')
    else:
        data = load_json('CTDRED/test.json')
        rel_to_id = load_json('CTDRED/relation_to_id.json')
        result_nod = load_json('checkpoint/ctd_finetune_lr02/result.json')
        result_wod = load_json('checkpoint/test_ctd_finetune_t0705_lr02/result.json')
    label_set = set()
    title2id = {}
    id2item = {}
    for item in data:
        title = item['title']
        assert title not in title2id
        tid = len(title2id)
        title2id[title] = tid
        id2item[tid] = item
        for lab in item['labels']:
            label_set.add((tid, lab['h'], lab['t'], rel_to_id[lab['r']]))
    instance, predict = len(label_set), len(result_nod)
    correct = 0
    set_nod = set()
    for item in result_nod:
        tup = title2id[item['title']], item['h_idx'], item['t_idx'], rel_to_id[item['r']]
        if tup in label_set:
            correct += 1
        set_nod.add(tup)
    pre, rec = correct / predict, correct / instance
    print(f'correct: {correct} predict: {predict} instance: {instance}')
    print(f'precision: {pre} recall: {rec} f1: {2 * pre * rec / (pre + rec)}')

    filter_set = copy.deepcopy(label_set)
    predict_wd, correct_wd = len(result_wod), 0
    set_wd = set()
    for item in result_wod:
        tup = title2id[item['title']], item['h_idx'], item['t_idx'], rel_to_id[item['r']]
        if tup in label_set:
            correct_wd += 1
            filter_set.discard(tup)
        set_wd.add(tup)
    pre_wd, rec_wd = correct_wd / predict_wd, correct_wd / instance
    print(f'correct: {correct_wd} predict: {predict_wd} instance: {instance}')
    print(f'precision: {pre_wd} recall: {rec_wd} f1: {2 * pre_wd * rec_wd / (pre_wd + rec_wd)}')

    print(len(set_nod & label_set), len(set_wd & label_set))
    # 3605 3332 2700 905 632
    intersect_set = set_nod & set_wd
    pre_set = set_nod - intersect_set
    suf_set = set_wd - intersect_set
    print(len(set_nod), len(set_wd), len(intersect_set), len(pre_set), len(suf_set))

    # filter which are in label_set
    filter_set = set()
    for item in intersect_set:
        if item in label_set:
            filter_set.add(item)
    intersect_set = filter_set

    filter_set = set()
    for item in pre_set:
        if item in label_set:
            filter_set.add(item)
    pre_set = filter_set

    filter_set = set()
    for item in suf_set:
        if item in label_set:
            filter_set.add(item)
    suf_set = filter_set
    # 2118 338 172
    print(len(intersect_set), len(pre_set), len(suf_set))
    exit()

    print('------ predict by model')
    tup = random.choice(list(filter_set))
    print(tup)
    tid, h, t, rid = tup
    item = id2item[tid]
    for idx, sent in enumerate(item['sents']):
        print(idx, ' '.join(sent))
    print_json(item['labels'])
    print_json(item['vertexSet'][h])
    print_json(item['vertexSet'][t])

    """
    {
        "name": "colon carcinogenesis",
        "pos": [
            12,
            14
        ],
        "sent_id": 10,
        "type": "Disease"
    } -- 2

    {
        "name": "n-tritriacontane-16, 18-dione",
        "pos": [
            15,
            17
        ],
        "sent_id": 0,
        "type": "Chemical"
    },
    {
        "name": "n-tritriacontane-16, 18-dione",
        "pos": [
            16,
            18
        ],
        "sent_id": 1,
        "type": "Chemical"
    } -- 7

    {
        "evidence": [],
        "h": 7,
        "r": "chem_disease_therapeutic",
        "t": 2
    },
    
    0 Modifying effects of the naturally occurring antioxidants gamma-oryzanol , phytic acid , tannic acid and 
      n-tritriacontane-16, 18-dione in a rat wide-spectrum organ carcinogenesis model.
    1 The modifying effects of the naturally occurring antioxidants gamma-oryzanol , phytic acid , tannic acid and 
      n-tritriacontane-16, 18-dione (TTAD) were investigated in a rat wide-spectrum organ carcinogenesis model.
    2 Animals were initiated with two i.p.
    3 injections of 1000 mg/kg body wt 2,2'-dihydroxy-di-n-propylnitrosamine ( DHPN ) followed by two i.g.
    4 administrations of 1500 mg/kg body wt N-ethyl-N-hydroxy-ethylnitrosamine (EHEN), and then three s.c.
    5 injections of 75 mg/kg body wt 3,2'-dimethyl-4-aminobiphenyl (DMAB) during the first 3 weeks.
    6 Starting 1 week after the last injection, groups of rats received diet containing 1% gamma-oryzanol , 2% phytic 
      acid , 0.2% TTAD or 1% tannic acid or basal diet alone for 32 weeks.
    7 Animals were then killed and complete autopsy was performed at the end of week 36.
    8 Histological examination revealed enhancement of lung carcinogenesis by gamma-oryzanol , and the incidence of 
      urinary bladder papillomas to be increased by phytic acid .
    9 On the other hand, TTAD inhibited hepatic and pancreatic carcinogenesis .
    10 Phytic acid and tannic acid were marginally effective in inhibiting hepatic and colon carcinogenesis 
       respectively.
    11 The results thus indicated that naturally occurring antioxidants each exert specific modifying effects 
       depending on the organ site and indicate that wide-spectrum carcinogenesis models are useful for 
       defining complex influences.
    
    (4191, 3, 2, 1)
    ------
    (4191, 5, 2, 0)
    (4191, 5, 4, 1)
    0 A seizure , and electroencephalographic signs of a lowered seizure threshold, associated with fluvoxamine 
      treatment of obsessive-compulsive disorder .
    1 A 38-year-old patient with severe obsessive-compulsive disorder received fluvoxamine in a clinical study.
    2 Psychometric ratings showed marked clinical improvement in the third week of fluvoxamine administration, but 
      after 8 weeks, at a dose of 300 mg per day, he suffered a grand mal seizure after drinking a glass of beer 
    (0.2 liter).
    3 He had no history of previous epileptic seizures .
    4 Careful neurological evaluation including computer tomography and magnetic resonance imaging of the brain 
      revealed no signs of acute disease .
    5 EEG before the fit did not show epileptiform activity ; after the fit, spikes and spike-wave complexes appeared, 
      which disappeared upon discontinuation of fluvoxamine .
    6 Since his obsessive-compulsive symptoms had responded well to fluvoxamine and worsened after its discontinuation,
    the drug was cautiously reintroduced.
    7 Improvement of the obsessive-compulsive symptoms was observed again, but spikes and spike-wave complexes 
      reappeared at a dose of 50 mg per day.
    8 Under anticonvulsant treatment with carbamazepine , fluvoxamine was increased to 100 mg per day.
    9 No seizures occurred during a follow-up to two years.
    [
        {
            "evidence": [], 
            "h": 3, 
            "r": "chem_disease_therapeutic", 
            "t": 2
        }, 
        {
            "evidence": [], 
            "h": 5, 
            "r": "chem_disease_therapeutic", 
            "t": 4
        }, 
        {
            "evidence": [], 
            "h": 5, 
            "r": "chem_disease_marker/mechanism", 
            "t": 2
        }
    ]
    [
        {
            "name": "carbamazepine", 
            "pos": [
                4, 
                5
            ], 
            "sent_id": 8, 
            "type": "Chemical"
        }
    ]
    [
        {
            "name": "seizure", 
            "pos": [
                1, 
                2
            ], 
            "sent_id": 0, 
            "type": "Disease"
        }, 
        {
            "name": "seizure", 
            "pos": [
                9, 
                10
            ], 
            "sent_id": 0, 
            "type": "Disease"
        }, 
        {
            "name": "seizure", 
            "pos": [
                30, 
                31
            ], 
            "sent_id": 2, 
            "type": "Disease"
        }, 
        {
            "name": "seizures", 
            "pos": [
                1, 
                2
            ], 
            "sent_id": 9, 
            "type": "Disease"
        }
    ]
    """
