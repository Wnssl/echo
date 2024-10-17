import sys

import numpy as np
import torch
import json
from tqdm import tqdm
from process import get_preprocessed_tensor
import matplotlib.pyplot as plt
import datetime
import os
import argparse
import csv
from sklearn.metrics import r2_score
from utils import *

def load_model(model_path):
    # 加载模型
    model =None

    if model_path.endswith('.pt'):
        try:
            model = torch.load(model_path).cuda()
            model.eval()
            print(model)
        except Exception as e:
            print('模型加载错误')
            print(e)
    elif model_path.endswith('.pth'):
        # # 导入模型
        # model = ()
        # model.load_state_dict(torch.load(model_path)
        model = None
        print('需要手动进行完善')
        sys.exit(1)
    else:
        print('模型路径错误')
        sys.exit(2)

    return model

def predict(args):
    """ 1 """
    frame_n = args['nums']
    is_multi_circle = args['multi circles']
    type = args['type']
    test_path = args['path']
    model_path = args['model']
    model_name = args['model_name']
    mask = args['mask']
    output_path = make_dir(type)
    print(f"video frames for predict : {frame_n}")
    print(f"Multi Circle predict: {is_multi_circle }")
    print(f"predict data type {type}")
    print(f"output files path : {output_path}")
    # 加载模型
    print(f"model path：{model_path}")
    print(f"model name : {model_name}")
    # 加载参数

    model = load_model(model_path)

    # 加载dict 存储未预测的基础信息
    with open(os.path.join(output_path, 'validation.json'), 'r+') as f:
        data_dict = json.load(f)


    with open(os.path.join(output_path, 'result.txt'), 'w+') as f:
        f.write('start eval------')
        f.write("args below")
        for k, v in args.items():
            f.write(f"{k} : {v}\n")
        f.write('-------------------\n')
    # 测试模型

    with open(os.path.join(output_path, 'result.txt'), 'a+') as file_writer:

        with torch.no_grad():
            if is_multi_circle:
                print('多周期预测')
                files = get_test_files(test_path)
                with torch.no_grad():
                    for f in tqdm(files):
                        assert f.endswith('.dcm')

                        try:
                            tensor = get_preprocessed_tensor(f, frame_n)
                        except:
                            print("no enough frames")
                            continue
                        tensor = tensor.permute(0, 2, 1, 3, 4).to(torch.float32)
                        # print(tensor.shape)
                        tensor = tensor.cuda()
                        outputs = model(tensor)
                        print(outputs)
                        mean = torch.mean(outputs)
                        # print(mean)
                        output = mean.cpu().numpy()
                        # print(output)
                        names = f.split('/')
                        # print(names)
                        key = names[-1].split('.')[0]
                        data_dict[key]['predict'] = float(output)
                        data_dict[key]['AE'] = float(data_dict[key]['RVEF'] - output)
                        values = [str(i) for i in outputs.cpu().numpy()]
                        file_writer.write(f"{key} : all values-{values} mean-{mean} ground-truth-{data_dict[key]['RVEF']} \n")

            else:
                print('单周期预测')
                files = get_test_files(test_path)
                with torch.no_grad():
                    for f in tqdm(files):
                        assert f.endswith('.npz')
                        try:
                            array = np.load(f)
                            tensor = torch.tensor(array['arr']).unsqueeze(0)
                            # print(tensor.shape)

                        except Exception as e:
                            print(e)
                            continue

                        tensor = tensor.permute(0, 2, 1, 3, 4).to(torch.float32)
                        # print(tensor.shape)
                        tensor = tensor.cuda()
                        outputs = model(tensor)
                        print(outputs)
                        mean = torch.mean(outputs)
                        # print(mean)
                        output = mean.cpu().numpy()
                        # print(output)
                        names = f.split('/')
                        # print(names)
                        key = names[-1].split('.')[0]
                        data_dict[key]['predict'] = float(output)
                        data_dict[key]['AE'] = float(data_dict[key]['RVEF'] - output)
                        values = [str(i) for i in outputs.cpu().numpy()]
                        file_writer.write(
                            f"{key} : all values-{values} mean-{mean} ground-truth-{data_dict[key]['RVEF']} \n")


    # 保存验证结果
    with open(os.path.join(output_path, 'validation.json'), 'w+') as f:
            # print(data_dict)
            json.dump(data_dict, f)

    # 根据上述结果生成统计信息
    if type == 'validation':
        count(os.path.join(output_path, 'validation.json'), output_path)
        analyze(os.path.join(output_path, 'validation.json'), output_path)

    else:
        count(os.path.join(output_path, 'train.json'), output_path)
        analyze(os.path.join(output_path, 'train.json', output_path))

    # 将模型保存到ouput内
    os.rename(model_path, os.path.join(output_path, 'model.pt'))


def count(path, out_path):
    """ 计算mAE, mSE 指标"""
    rvefs = []
    predicts = []

    with open(path, 'r+') as f:
        data_dict = json.load(f)
        # print(data_dict)
        total_AE = 0.0
        total_SE = 0.0
        count = 0
        for k, v in data_dict.items():
            try:
                total_AE += abs(v['AE'])
                total_SE += v['AE']**2
                rvefs.append(v['RVEF'])
                predicts.append(v['predict'])
                count +=1
            except:
                print("no predict values !!!")
                pass
        predicts = np.array(predicts)
        rvefs = np.array(rvefs)
        rmse = np.sqrt(np.mean((predicts - rvefs) ** 2))
        r2 = r2_score(rvefs, predicts)
        print(f'mAE:{total_AE / count}')
        print(f'mSE:{total_SE / count}')
        print(f'RMSE:{rmse}')
        print(f'r2:{r2}')
        print(count)
        with open(os.path.join(out_path, 'result.txt'), 'a+') as writer:
            writer.write('--------------------------\n')
            writer.write(f'mAE:{total_AE / count}\n')
            writer.write(f'mSE:{total_SE / count}\n')
            writer.write(f'RMSE:{rmse}\n')
            writer.write(f"r2:{r2}\n")
            writer.write(str(count)+'\n')


def analyze(dict_path, output_path):
    data_json = json.load(open(dict_path, 'r+'))
    AEs = []
    predicts = []
    rvefs = []
    for k, v in data_json.items():
        try:
            if v['AE'] is not None:
                AEs.append(v['AE'])
                predicts.append(v['predict'])
                rvefs.append(v['RVEF'])
        except:
            print("no predict values !!!")
            pass

    # 绘制直方图
    print(AEs)
    plt.figure(1)
    plt.hist(AEs, bins=70, color='skyblue', edgecolor='black')
    # 添加标题和标签
    plt.title('Histogram of ALL RVEF DATA')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    # 保存图像
    plt.savefig(os.path.join(output_path, 'result1.png'))
    # 显示图形
    plt.show()

    # 绘制散点图
    plt.figure(2)
    plt.scatter(rvefs, predicts, s=3, color='blue', label='Scatter Plot')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='y=x Line')
    plt.xlabel('ground-truth')
    plt.ylabel('predict')
    plt.title('Scatter Plot')
    plt.legend()
    plt.xlim(0.1, 0.8)
    plt.ylim(0.1, 0.8)
    plt.savefig(os.path.join(output_path, 'result2.png'))
    plt.show()

def get_args():
    """ 预测需要的参数 """
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help='num_of_frames_to_predict', default=32)
    parser.add_argument('--m', type=bool, help='mean  RVEF', default=False)
    parser.add_argument('--t', type=str, help='validation or train',choices=['validation', 'train'], default='validation')
    # 存储待预测数据位置
    parser.add_argument('--p', type=str, help='data path', default='test_data_demo/')
    parser.add_argument('--model', type=str, help='model path', default='input/best_mode045.pt')
    parser.add_argument('--model_name', type=str, help='model name', default='resnet3d')
    parser.add_argument('--mask', type=bool, help='mask used', default=True)
    # parser.set_defaults(augment=True)
    args = parser.parse_args()

    return args



def main():
    """ predict 主函数"""
    args = get_args()
    args_dict = {
        'nums' : args.n,
        'multi circles' : args.m,
        'type' : args.t,
        'path' : args.p,
        'model': args.model,
        'model_name': args.model_name,
        'mask' : args.mask,
    }
    print(args_dict)

    predict(args_dict)


if __name__ == '__main__':
    # make_dir('train')
    main()

