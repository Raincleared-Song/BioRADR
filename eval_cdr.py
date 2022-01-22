import os
import json
import argparse


def gen_train_facts(data_file_name):
    fact_file_name = data_file_name.replace('.gold', '.fact')

    if os.path.exists(fact_file_name):
        facts = set()
        triples = json.load(open(fact_file_name, 'r', encoding='utf-8'))
        for triple in triples:
            facts.add(tuple(triple))
        return facts

    facts = set()
    ori_data = [line.strip().split('|') for line in open(data_file_name, 'r', encoding='utf-8').readlines()]
    for dd in ori_data:
        if dd[4] != '1:CID:2':
            continue
        h, t = dd[1], dd[2]
        facts.add((h, t))
    json.dump(list(facts), open(fact_file_name, 'w', encoding='utf-8'))
    return facts


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-r', '--result_file', type=str, required=True, help='the result file path')
    arg_parser.add_argument('-o', '--output_file', type=str, required=True, help='the output file path')
    arg_parser.add_argument('-d', '--dev_as_test', action='store_true')
    args = arg_parser.parse_args()
    input_filename = args.result_file
    output_filename = args.output_file

    train_path = 'CDR/train_filter.gold'
    dev_path = 'CDR/dev_filter.gold'
    test_path = 'CDR/dev_filter.gold' if args.dev_as_test else 'CDR/test_filter.gold'
    test_doc_path = 'CDR/dev_cdr.json' if args.dev_as_test else 'CDR/test_cdr.json'

    fact_in_train = gen_train_facts(train_path)

    truth = set()
    truth_data = [line.strip().split('|') for line in open(test_path, 'r', encoding='utf-8').readlines()]
    for data in truth_data:
        if data[4] != '1:CID:2':
            continue
        pmid, h, t = data[0], data[1], data[2]
        truth.add((pmid, h, t))

    test_docs = json.load(open(test_doc_path, 'r', encoding='utf-8'))
    title2doc = {str(doc['pmid']): doc for doc in test_docs}

    answers = json.load(open(input_filename, 'r', encoding='utf-8'))
    correct_num, instance_num, predict_num, correct_in_train = 0, len(truth), 0, 0
    for pre in answers:
        title, h, t, r = pre['title'], pre['h_idx'], pre['t_idx'], pre['r']
        if r != 'Pos':
            continue
        doc = title2doc[title]
        h_cids = doc['cids'][h].split('|')
        t_cids = doc['cids'][t].split('|')
        for hid in h_cids:
            for tid in t_cids:
                triple = doc['pmid'], hid, tid
                predict_num += 1
                if triple in truth:
                    correct_num += 1
                    if (hid, tid) in fact_in_train:
                        correct_in_train += 1

    out_file = open(output_filename, 'w', encoding='utf-8')
    print(f'correct: {correct_num} instance: {instance_num} predict: {predict_num} in_train: {correct_in_train}')
    out_file.write(f'correct: {correct_num} instance: {instance_num} predict: {predict_num} '
                   f'in_train: {correct_in_train}')
    precision = correct_num / predict_num
    recall = correct_num / instance_num
    f1 = 2 * precision * recall / (precision + recall)
    print(f'RE_PRECISION: {round(precision * 100, 2)}\n'
          f'RE_RECALL: {round(recall * 100, 2)}\n'
          f'RE_F1: {round(f1 * 100, 2)}')
    out_file.write(f'RE_PRECISION: {round(precision * 100, 2)}\n'
                   f'RE_RECALL: {round(recall * 100, 2)}\n'
                   f'RE_F1: {round(f1 * 100, 2)}')

    precision = correct_num / predict_num
    ing_precision = (correct_num - correct_in_train) / (predict_num - correct_in_train)
    recall = correct_num / instance_num
    f1 = 2 * precision * recall / (precision + recall)
    ing_f1 = 2 * ing_precision * recall / (ing_precision + recall)
    print(f'RE_ignore_PRECISION: {round(ing_precision * 100, 2)}\n'
          f'RE_ignore_RECALL: {round(recall * 100, 2)}\n'
          f'RE_ignore_F1: {round(ing_f1 * 100, 2)}')
    out_file.write(f'RE_ignore_PRECISION: {round(ing_precision * 100, 2)}\n'
                   f'RE_ignore_RECALL: {round(recall * 100, 2)}\n'
                   f'RE_ignore_F1: {round(ing_f1 * 100, 2)}')
    out_file.close()
