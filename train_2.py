import argparse
from dataset import train_data, test_data, train_data_rnc
import torch
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from losses import *
from loss import RnCLoss
from tqdm import tqdm
from uniformer import get_feature_extractor
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='/tf_logs')


def train_one_epoch(model, regressor, device, optimizer, optimizer_regressor, criterion, scheduler, epoch, train_loader,
                    weighted=False):
    model.eval()
    regressor.train()

    criterion_mse = torch.nn.L1Loss()

    # 无用total_loss
    total_loss = 0.0

    for idx, batch in tqdm(enumerate(train_loader)):
        views1, _ = batch
        images = views1[0]
        labels = views1[1]
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        with torch.no_grad():
            features = model(images)

        features = features.detach()
        y_preds = regressor(features).squeeze(1)
        loss_reg = criterion_mse(y_preds, labels)

        optimizer_regressor.zero_grad()
        loss_reg.backward()
        optimizer_regressor.step()

        total_loss += loss_reg.item()
        if idx % 10 == 0:
            print(f'第{epoch}个epoch 第{idx} batch 的损失为 {loss_reg.item()}')

    return total_loss / len(train_loader)

def val_one_epoch(model, regressor, device, criterion, epoch, validation_loader):
    model.eval()
    regressor.eval()

    criterion_mse = torch.nn.L1Loss()

    # 无用total_loss
    total_loss = 0.0

    for idx, batch in tqdm(enumerate(validation_loader)):
        views1, _ = batch
        images = views1[0]
        labels = views1[1]
        bsz = labels.shape[0]

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)

        with torch.no_grad():
            features = model(images)
        features = features.detach()
        y_preds = regressor(features).squeeze(1)
        loss_reg = criterion_mse(y_preds, labels)


        total_loss += loss_reg.item()
        if idx % 10 == 0:
            print(f'第{epoch}个epoch 第{idx} batch 的损失为 {loss_reg.item()}')

    return total_loss / len(validation_loader)



def test_one_epoch(model, regressor, device, criterion, epoch, validation_loader):
    # for test
    model.eval()
    regressor.eval()
    predict_total_loss = 0.0
    print('开始验证 : --------')
    for batch_idx, (data, target) in tqdm(enumerate(validation_loader)):
        # 将输入数据和标签移动到GPU
        data, target = data.to(device), target.to(device)
        features = model(data)
        outputs = regressor(features)
        # print(outputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, target)
        predict_total_loss += loss.item()
    avg_loss_val = predict_total_loss / len(validation_loader)
    print(f'epoch{epoch} avg loss:{avg_loss_val}')

    return avg_loss_val


def my_collate_fn(batch):
    """ 最后返回填充后的数据， 标签， 对应的修改size"""
    # 获取batch中每个样本的尺寸
    # 数据 RVEF值 周期数
    data = [item[0] for item in batch]  # 提取出每个样本的数据
    target = [item[1] for item in batch]  # 提取出每个样本的标签

    data = torch.stack(data)  # 将数据堆叠成一个张量
    target = torch.tensor(target)
    target = target.unsqueeze(1)
    a, b, c, d, e = data.shape
    data = data.view(-1, c, d, e)

    return data, target


def main():
    parser = argparse.ArgumentParser()

    # train parser
    parser.add_argument('--batch_size', type=int, help='batch_size', default=16)
    parser.add_argument('--epoch_nums', type=int, help='epoch nums', default=50)
    parser.add_argument('--adam_learning_rate', type=float, help='learning rate', default=0.0001)
    parser.add_argument('--SGD_learning_rate', type=float, help='SGD learning rate', default=0.02)
    parser.add_argument('--momentum', type=float, help='SGD momentum', default=0.8)
    parser.add_argument('--optim', type=str, help='optimizer', choices=['SGD', 'Adam'], default='Adam')
    parser.add_argument('--criterion', type=str, help=' loss ', choices=['L1', 'MSE'], default='L1')
    parser.add_argument('--scheduler', type=str, help='lr scheduler', default='StepLR')
    parser.add_argument('--step_num', type=int, help='step num', default=15)
    parser.add_argument('--gamma', type=float, help='step gammer', default=0.1)
    parser.add_argument('--num_workers', type=int, help='cpu workers', default=15)
    parser.add_argument('--collate_fn', type=bool, help='work om collate_fn', default=False)
    parser.add_argument('--multi_criterion', type=str, help='if on multi loss function', default=True)
    parser.add_argument('--criterion1', type=str, help='loss for train',
                        choices=['L1', 'MSE', 'Focal L1', 'Focal MSE', 'diy'], default='diy')
    parser.add_argument('--random_seed', type=bool, help='if use random seed', default=True)
    parser.add_argument('--seed', type=int, help='random seed', default=666)
    # dataset parser
    parser.add_argument('--train_file_path', type=str, help='train files path', default='train_data_32')
    parser.add_argument('--test_file_path', type=str, help='test files path', default='test_data_32')
    parser.add_argument('--process_type', type=str, help='how to process data a :None  b : re-sampling '
                                                         'c : reweight', default='a')
    parser.add_argument('--input_dim_transpose', type=bool, help='if transpose data', default=True)
    parser.add_argument('--class_nums', type=int, help='if re-sampling then nums of class', default=8)
    parser.add_argument('--codebook_path', type=str, help='codebook path', default='codebook.csv')
    parser.add_argument('--reweight', type=str, help='how to reweight data', choices=['sqrt_inv', 'inverse', 'none'],
                        default='sqrt_inv')
    parser.add_argument('--lds', type=bool, help='decide if on lds', default=False)
    parser.add_argument('--lds_kernel', type=str, help='lds kernel', default='gaussian')
    parser.add_argument('--lds_ks', type=int, help='lds kernel size', default=5)
    parser.add_argument('--lds_sigma', type=int, help='lds sigma', default=2)
    # predict best model parser
    parser.add_argument('--n', type=int, help='num_of_frames_to_predict', default=32)
    parser.add_argument('--m', type=bool, help='mean  RVEF', default=False)
    parser.add_argument('--t', type=str, help='validation or train', choices=['validation', 'train'],
                        default='validation')
    # 存储待预测数据位置
    parser.add_argument('--p', type=str, help='data path', default='test_data_32/')
    parser.add_argument('--model', type=str, help='model path', default='input/uniformer_small.pt')
    parser.add_argument('--model_name', type=str, help='model name', default='uniformer_small')

    # model parser
    parser.add_argument('--fds', action='store_true', default=False, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=0, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=1, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=100, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=0, choices=[0, 3],
                        help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')

    # RnCLoss Parameters
    parser.add_argument('--rnc', type=bool, default=True, help='use RnC - loss')
    parser.add_argument('--temp', type=float, default=2, help='temperature')
    parser.add_argument('--label_diff', type=str, default='l1', choices=['l1'], help='label distance function')
    parser.add_argument('--feature_sim', type=str, default='l2', choices=['l2'], help='feature similarity function')

    # parser.add_argument()
    args = parser.parse_args()
    print(args)
    # 加载训练集
    train = train_data_rnc(args.train_file_path, isTrain=True, process_type=args.process_type,
                           input_dim_transpose=args.input_dim_transpose, class_nums=args.class_nums,
                           codebook_path=args.codebook_path, reweight=args.reweight, lds=args.lds,
                           lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)

    train, val = random_split(train, [len(train) - 400, 400])

    # 加载验证集
    test = test_data(args.test_file_path, isTrain=False, process_type=args.process_type,
                     input_dim_transpose=args.input_dim_transpose, class_nums=args.class_nums,
                     codebook_path=args.codebook_path, reweight=args.reweight, lds=args.lds,
                     lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.collate_fn:
        my_collate = my_collate_fn()
    else:
        my_collate = None

    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=my_collate)
    validation_loader = DataLoader(dataset=val, batch_size=args.batch_size, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=my_collate)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             collate_fn=my_collate)

    # model
    print('buliding model-----')
    # model = get_feature_extractor()
    # model = resnet18_3d(fds=args.fds, bucket_num=args.bucket_num, bucket_start=args.bucket_start,
    #                  start_update=args.start_update, start_smooth=args.start_smooth,
    #                  kernel=args.fds_kernel, ks=args.fds_ks, sigma=args.fds_sigma, momentum=args.fds_mmt)
    # model = resnet18_3d(3, include_top=True)
    # model = model.to(device)
    # model = RMT_S(None)
    # dim_in = model.head.in_features
    # regressor = get_shallow_mlp_head(dim_in, 1)
    # model = model.to(device)
    # regressor = regressor.to(device)
    from bkConv import final_model2
    # output_dim = 640
    # model = final_model2(output_dim=output_dim, include_top=False)
    from regressor import get_shallow_mlp_head
    # regressor = get_shallow_mlp_head(output_dim, 1)
    model = torch.load('best_model.pt')
    regressor = torch.load('best_regressor.pt')

    model = model.to(device)
    regressor = regressor.to(device)

    # loss function
    if args.multi_criterion:
        # train loss
        if args.criterion1 == 'L1':
            criterion1 = weighted_l1_loss
        elif args.criterion1 == 'MSE':
            criterion1 = weighted_mse_loss
        elif args.criterion1 == 'Focal_L1':
            criterion1 = weighted_focal_l1_loss
        elif args.criterion1 == 'Focal_MSE':
            criterion1 = weighted_focal_mse_loss
        else:
            # 自定义loss
            if args.criterion1 == 'diy':
                if args.rnc:
                    criterion1 = RnCLoss(temperature=args.temp, label_diff=args.label_diff,
                                         feature_sim=args.feature_sim)
            else:
                pass
        # test loss
    if args.criterion == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # optim
    if args.optim == 'Adam':
        optimizer = optim.AdamW(model.parameters(), lr=args.adam_learning_rate)
        optimizer_reg = optim.AdamW(regressor.parameters(), lr=args.adam_learning_rate)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.SGD_learning_rate, momentum=args.momentum)
    else:
        optimizer = None

    # optim step
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_num, gamma=args.gamma)
        scheduler_reg = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_num, gamma=args.gamma)
    else:
        # 自定义scheduler
        scheduler = None
        pass

    # 训练是否加权
    weighted = False
    if args.process_type == 'c':
        weighted = True

    # best result  ---- used to print and save best model
    best_result = 9999

    # 用于保存 val 结果
    results = []

    # def train_one_epoch(model, regressor, device, optimizer, optimizer_reg, criterion, scheduler,  train_loader, weighted=False):
    fds = args.fds
    start_update = args.start_update
    for epoch in range(args.epoch_nums):
        # for train
        if args.multi_criterion:
            train_loss = train_one_epoch(model, regressor, device, optimizer, optimizer_reg, criterion1, scheduler,
                                         epoch, train_loader)
        else:
            train_loss = train_one_epoch(model, regressor, device, optimizer, optimizer_reg, criterion, scheduler,
                                         epoch, train_loader, fds, start_update, weighted)

        # for test
        avg_loss_val = val_one_epoch(model, regressor, device, criterion, epoch, validation_loader)
        scheduler_reg.step()
        writer.add_scalar('test_loss', avg_loss_val, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        # results.append(avg_loss_val)
        # 输出最好结果
        if avg_loss_val < best_result:
            best_result = avg_loss_val
            best_train_loss = train_loss
            best_regressor = regressor
            best_epoch = epoch
            best_model = model

        print(f'当前最佳模型{best_epoch}  val : {best_result}')

    # 对最好模型进行保存
    # print(results)
    torch.save(best_model, f'best_model.pt')
    torch.save(best_regressor, f'best_regressor.pt')

    test_loss = test_one_epoch(best_model, best_regressor, device, criterion, 0, test_loader)
    print(f'test loss : {test_loss}')


if __name__ == '__main__':
    main()
