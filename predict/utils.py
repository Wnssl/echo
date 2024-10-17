# ****************
# 工具方法
import os
import csv
import json
import datetime

def get_test_files(file_path):
    files = []
    for f in os.listdir(file_path):
        files.append(file_path+f)

    return files

def make_json(file_path, type='validation'):
    """ 制作test 和"""
    book = 'input/codebook.csv'
    test_dict = {}
    # 待保存的json字典
    reader = csv.reader(open(book,'r'))
    next(reader)
    for line in reader:
        if line[-1] == type:
            test_dict[line[0]] = {}
            test_dict[line[0]]['RVEF'] = float(line[-2])
            test_dict[line[0]]['RVESV'] = float(line[-3])
            test_dict[line[0]]['RVEDV'] = float(line[-4])
            test_dict[line[0]]['AE'] = None
            test_dict[line[0]]['predict'] = None
            test_dict[line[0]]['view_type'] = line[-7]
    with open(file_path, 'w+') as json_file:
        # print(test_dict)
        json.dump(test_dict, json_file)
        print(f"完成生成{file_path}")


def make_dir(type='validation'):
    """ 生成文件夹  同时生成需要的json文件  用于保存输出结果"""

    output_path = 'output'

    time = datetime.datetime

    times = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')

    if os.path.exists(os.path.join(output_path, times)):
        print('文件夹已经存在')
        dir = os.path.join(output_path, times)
        # assert 1 < 0
    else:
        os.mkdir(os.path.join(output_path, times))
        dir = os.path.join(output_path, times)

    if type == 'validation':
        file_path = os.path.join(dir, 'validation.json')
        print(file_path)
        make_json(file_path, type)

    else:
        file_path = os.path.join(dir, 'train.json')
        print(file_path)
        make_json(file_path, type)

    return dir

