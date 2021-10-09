import json
import os
import argparse


def gen_train_facts(data_file_name):
    fact_file_name = data_file_name.replace('.json', '.fact')

    if os.path.exists(fact_file_name):
        fact_in_train = set()
        triples = json.load(open(fact_file_name))
        for triple in triples:
            fact_in_train.add(tuple(triple))
        return fact_in_train

    fact_in_train = set()
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        entities = data['vertexSet']
        for lab in data['labels']:
            rel = lab['r']
            for m1 in entities[lab['h']]:
                for m2 in entities[lab['t']]:
                    fact_in_train.add((m1['name'], m2['name'], rel))
    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-r', '--result_file', type=str, required=True, help='the result file path')
    arg_parser.add_argument('-o', '--output_file', type=str, required=True, help='the output file path')
    arg_parser.add_argument('-d', '--dev_as_test', action='store_true')
    args = arg_parser.parse_args()
    input_filename = args.result_file
    output_filename = args.output_file

    pretrain_path = 'CTDRED/pretrain_mixed.json'
    train_path = 'CTDRED/train_mixed.json'
    dev_path = 'CTDRED/dev.json'
    test_path = 'CTDRED/dev.json' if args.dev_as_test else 'CTDRED/test.json'
    # train_path = 'CTDRED/cdr_train.json'
    # dev_path = 'CTDRED/cdr_dev.json'
    # test_path = 'CTDRED/cdr_dev.json' if args.dev_as_test else 'CTDRED/cdr_test.json'
    # train_path = 'Chemprot/chemprot_train_sent_fr.json'
    # dev_path = 'Chemprot/chemprot_dev_sent_fr.json'
    # test_path = 'Chemprot/chemprot_dev_sent_fr.json' if args.dev_as_test else 'Chemprot/chemprot_test_sent_fr.json'
    to_test = test_path

    fact_in_train_annotated = gen_train_facts(train_path)
    fact_in_train_distant = gen_train_facts(pretrain_path)
    # fact_in_train_distant = set()

    output_file = open(output_filename, 'w')
    truth = json.load(open(to_test))

    std = set()
    title_set = set()
    title_to_vertex = {}

    for doc in truth:
        title = doc['title']
        title_set.add(title)

        vertexSet = doc['vertexSet']
        title_to_vertex[title] = vertexSet

        for label in doc['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std.add((title, r, h_idx, t_idx))

    tot_relations = len(std)

    tmp = json.load(open(input_filename))
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    submission_answer = [tmp[0]]
    for i in range(1, len(tmp)):
        a = tmp[i]
        b = tmp[i - 1]
        if (a['title'], a['h_idx'], a['t_idx'], a['r']) != (b['title'], b['h_idx'], b['t_idx'], b['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    title_set2 = set()
    for ans in submission_answer:
        title = ans['title']
        h_idx = ans['h_idx']
        t_idx = ans['t_idx']
        r = ans['r']
        title_set2.add(title)
        if title not in title_to_vertex:
            continue
        vertexSet = title_to_vertex[title]

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    print(f'correct_num: {correct_re} predict_num: {len(submission_answer)} instance_num: {tot_relations}')
    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p+re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re-correct_in_train_annotated) \
        / (len(submission_answer)-correct_in_train_annotated)
    re_p_ignore_train = 1.0 * (correct_re-correct_in_train_distant) \
        / (len(submission_answer) - correct_in_train_distant)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / \
                                       (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train+re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    print('RE_F1:', re_f1)
    print('RE_PRECISION:', re_p)
    print('RE_RECALL:', re_r)
    print('RE_ignore_annotated_F1:', re_f1_ignore_train_annotated)
    print('RE_ignore_distant_F1:', re_f1_ignore_train)

    output_file.write("RE_F1: %f\n" % re_f1)
    output_file.write('RE_PRECISION: %f\n' % re_p)
    output_file.write('RE_RECALL: %f\n' % re_r)
    output_file.write("RE_ignore_annotated_F1: %f\n" % re_f1_ignore_train_annotated)
    output_file.write("RE_ignore_distant_F1: %f\n" % re_f1_ignore_train)

    output_file.close()
