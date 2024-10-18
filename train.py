import argparse
from dataset1 import train_data, test_data
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from losses import *
from tqdm import tqdm
# from uniformer import get_feature_extractor
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter(log_dir='/tf_logs')
def train_one_epoch(model, device, optimizer, criterion, scheduler, epoch, train_loader, fds, start_update, weighted=False):

    # for train
    total_loss = 0.0
    model.train()

    if not weighted:
        for batch_idx, (input, target) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            input, target = input.to(device), target.to(device)
            # print(input.shape)
            outputs = model(input)
            # outputs = outputs.squeeze(1)
            # print(outputs.shape)
            mean = torch.mean(outputs)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'第{epoch}个epoch 第{batch_idx} batch 的损失为 {loss.item()}')
                # writer.add_scalar('output', mean, epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('total_loss', total_loss, epoch * len(train_loader) + batch_idx)
        # torch.save(model, f'cnn/cnn_{epoch}.pt')
        avg_loss = total_loss / len(train_loader)
        # writer.add_scalar('mAE', avg_loss, epoch)
        print(f"epoch : {epoch} mAE为{avg_loss}")

    else:
        for batch_idx, (data, target, weight) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            data, target, weight = data.to(device), target.to(device), weight.to(device)
            # print(input.shape)
            if fds:
                outputs, _ = model(data, target, epoch)
            else:
                outputs = model(data, target, epoch)
            outputs = outputs.squeeze(1)
            # print(outputs.shape)
            mean = torch.mean(outputs)
            # print(mean)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                print(f'第{epoch}个epoch 第{batch_idx} batch 的损失为 {loss.item()}')
                # writer.add_scalar('output', mean, epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('loss', loss.item(), epoch * len(train_loader) + batch_idx)
                # writer.add_scalar('total_loss', total_loss, epoch * len(train_loader) + batch_idx)
        # torch.save(model, f'cnn/cnn_{epoch}.pt')
        avg_loss = total_loss / len(train_loader)
        # writer.add_scalar('mAE', avg_loss, epoch)
        print(f"epoch : {epoch} mAE为{avg_loss}")

        if fds and epoch >= start_update:
            print(f"Create Epoch [{epoch}] features of all training data...")
            encodings, labels = [], []
            with torch.no_grad():
                for (inputs, targets, _) in tqdm(train_loader):
                    inputs = inputs.cuda(non_blocking=True)
                    # print(targets.shape)
                    outputs, feature = model(inputs, targets, epoch)
                    encodings.extend(feature.data.squeeze().cpu().numpy())
                    labels.extend(targets.data.squeeze().cpu().numpy())

            encodings, labels = torch.from_numpy(np.vstack(encodings)).cuda(), torch.from_numpy(
                np.hstack(labels)).cuda()
            model.FDS.update_last_epoch_stats(epoch)
            model.FDS.update_running_stats(encodings, labels, epoch)

    scheduler.step()
    return avg_loss

def test_one_epoch(model, device, criterion, epoch, validation_loader):
    # for test
    model.eval()
    predict_total_loss = 0.0
    print('开始验证 : --------')
    for batch_idx, (data, target, _) in tqdm(enumerate(validation_loader)):
        print(data.shape)
        print(target)
        # 将输入数据和标签移动到GPU
        data, target = data.to(device), target.to(device)
        outputs = model(data)
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
    parser.add_argument('--epoch_nums', type=int, help='epoch nums', default=100)
    parser.add_argument('--adam_learning_rate', type=float, help='learning rate', default=0.0003)
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
                        choices=['L1', 'MSE', 'Focal L1', 'Focal MSE', 'diy'], default='L1')
    parser.add_argument('--random_seed', type=bool, help='if use random seed', default=True)
    # parser.add_argument('--seed', type=int, help='random seed', default=666)
    # dataset parser
    parser.add_argument('--train_file_path', type=str, help='train files path', default='../train_video')
    parser.add_argument('--test_file_path', type=str, help='test files path', default='../test_video')
    parser.add_argument('--process_type', type=str, help='how to process data a :None  b : re-sampling '
                                                         'c : reweight', default='c')
    parser.add_argument('--input_dim_transpose', type=bool, help='if transpose data', default=False)
    parser.add_argument('--class_nums', type=int, help='if re-sampling then nums of class', default=7)
    parser.add_argument('--codebook_path', type=str, help='codebook path', default='codebook.csv')
    parser.add_argument('--reweight', type=str, help='how to reweight data', choices=['sqrt_inv', 'inverse', 'none'],
                        default='sqrt_inv')
    parser.add_argument('--lds', type=bool, help='decide if on lds', default=True)
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
    parser.add_argument('--fds', action='store_true', default=True, help='whether to enable FDS')
    parser.add_argument('--fds_kernel', type=str, default='gaussian',
                        choices=['gaussian', 'triang', 'laplace'], help='FDS kernel type')
    parser.add_argument('--fds_ks', type=int, default=5, help='FDS kernel size: should be odd number')
    parser.add_argument('--fds_sigma', type=float, default=1, help='FDS gaussian/laplace kernel sigma')
    parser.add_argument('--start_update', type=int, default=20, help='which epoch to start FDS updating')
    parser.add_argument('--start_smooth', type=int, default=21, help='which epoch to start using FDS to smooth features')
    parser.add_argument('--bucket_num', type=int, default=80, help='maximum bucket considered for FDS')
    parser.add_argument('--bucket_start', type=int, default=20, choices=[0, 3],
                        help='minimum(starting) bucket for FDS, 0 for IMDBWIKI, 3 for AgeDB')
    parser.add_argument('--fds_mmt', type=float, default=0.9, help='FDS momentum')
    
    #BMSE
    parser.add_argument('--bmse', action='store_true', default=False, help='use Balanced MSE')
    parser.add_argument('--imp', type=str, default='bni', choices=['gai', 'bmc', 'bni'], help='implementation options')
    parser.add_argument('--gmm', type=str, default='gmm.pkl', help='path to preprocessed GMM')
    parser.add_argument('--init_noise_sigma', type=float, default=10., help='initial scale of the noise')
    parser.add_argument('--sigma_lr', type=float, default=1e-2, help='learning rate of the noise scale')
    parser.add_argument('--balanced_metric', action='store_true', default=False, help='use balanced metric')
    parser.add_argument('--fix_noise_sigma', action='store_true', default=False, help='disable joint optimization')
    
    # parser.add_argument()
    args = parser.parse_args()
    print(args)
    # 加载训练集
    train = train_data(args.train_file_path, isTrain=True, process_type=args.process_type,
                       input_dim_transpose=args.input_dim_transpose, class_nums=args.class_nums,
                       codebook_path=args.codebook_path, reweight=args.reweight, lds=args.lds,
                       lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)

    train, validate = random_split(train, [len(train)-400, 400])
    print(len(train))
    print(len(validate))
    # 加载验证集
    test = test_data(args.test_file_path, isTrain=False, process_type=args.process_type,
                     input_dim_transpose=args.input_dim_transpose, class_nums=args.class_nums,
                     codebook_path=args.codebook_path, reweight=args.reweight, lds=args.lds,
                     lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)

    # cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataLoader
    
    if args.collate_fn:
        my_collate = my_collate_fn()
    else:
        my_collate = None
    
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                  collate_fn=my_collate)
    validation_loader = DataLoader(dataset=validate, batch_size=args.batch_size, shuffle=False,
                                       num_workers=args.num_workers,collate_fn=my_collate)
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
    # model =  SwinTransformer3D()
    from bkConv import final_model, final_model2
    model = final_model2()
    # model.head.bias.data[0] = 0.516

    model = model.to(device)

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
            if args.bmse:
                from balanced_mse import GAILoss, BMCLoss, BNILoss
                if args.imp == 'gai':
                    criterion1 = GAILoss(args.init_noise_sigma, args.gmm)
                elif args.imp == 'bmc':
                    criterion1 = BMCLoss(args.init_noise_sigma)
                elif args.imp == 'bni':
                    bucket_centers, bucket_weights = train.get_bucket_info(
                        max_target=76, lds=True, lds_kernel=args.lds_kernel, lds_ks=args.lds_ks, lds_sigma=args.lds_sigma)
                    criterion1 = BNILoss(args.init_noise_sigma, bucket_centers, bucket_weights)
            else:
                criterion1 = None
            pass
        # test loss
    if args.criterion == 'L1':
        criterion = nn.L1Loss()
    else:
        criterion = nn.MSELoss()

    # optim
    if args.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.adam_learning_rate)
    elif args.optim == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.SGD_learning_rate, momentum=args.momentum)
    else:
        optimizer = None

    # optim step
    if args.scheduler == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_num, gamma=args.gamma)
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

    #用于保存 val 结果
    results = []


    fds = args.fds
    print(fds)
    start_update = args.start_update
    for epoch in range(args.epoch_nums):
        # for train
        if args.multi_criterion:
            train_loss = train_one_epoch(model, device, optimizer, criterion1, scheduler, epoch, train_loader, fds, start_update, weighted)
        else:
            train_loss = train_one_epoch(model, device, optimizer, criterion, scheduler, epoch, train_loader,fds, start_update, weighted)

        #for test
        avg_loss_val = test_one_epoch(model, device, criterion, epoch, validation_loader)
        writer.add_scalar('test_loss', avg_loss_val, epoch)
        writer.add_scalar('train_loss', train_loss, epoch)
        results.append(avg_loss_val)
        #输出最好结果
        if avg_loss_val < best_result:
            best_result = avg_loss_val
            best_train_loss = train_loss
            best_epoch = epoch
            best_model = model

        print(f'当前最佳模型{best_epoch} : val : {best_result} train : {best_train_loss}')

    # 对最好模型进行保存
    print(results)
    torch.save(best_model, f'best_model.pt')
    model = torch.load('best_model.pt')
    
    model.eval()
    predict_total_loss = 0.0
    print('开始验证 : --------')
    for batch_idx, (data, target) in tqdm(enumerate(test_loader)):
        # 将输入数据和标签移动到GPU
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        # print(outputs)
        outputs = outputs.squeeze(1)
        loss = criterion(outputs, target)
        predict_total_loss += loss.item()
    avg_loss_val = predict_total_loss / len(test_loader)
    print(f'avg loss:{avg_loss_val}')
if __name__ == '__main__':
    main()
