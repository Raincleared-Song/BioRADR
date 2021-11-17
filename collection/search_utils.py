import json


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
