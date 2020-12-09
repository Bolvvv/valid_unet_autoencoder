import torch
from torch import utils
from dataset import CustomDataset
from config import TrainingConfig
from torchvision import datasets, transforms, models
import torch.nn as nn
import time
from unet import UNet
import sys
from PIL import Image

#设置cuda使用
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_valid(unet_model, densenet_model, config, data_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(unet_model.parameters(), lr=config.lr)
    result_file_path = './result/'+time.strftime('%m%d_%H%M_%S',time.localtime(time.time()))+'_results.csv'
    #将信息写入csv文件中
    with open(result_file_path, 'w') as f:
        f.write('batch_size %d, lr %f, epoches %d, start_time %s\n' % (config.batch_size, config.lr, config.epoches, time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))))
        f.write('epoch,train_loss,tgt_acc,src_trans_acc\n')

    unet_model.train()
    densenet_model.eval()
    src_trans_img_max_acc = 0#densenet预测的转换图片的最高准确率
    for epoch in range(config.epoches):
        total_step = len(data_loader)
        tgt_img_correct = 0
        src_trans_img_correct = 0
        total = 0#一个epoch的样本总数
        total_loss = 0#一个epoch的总loss
        for i, (src_img, tgt_img, label) in enumerate(data_loader):
            src_img = src_img.to(device)
            tgt_img = tgt_img.to(device)
            label = label.to(device)
            #forward
            output = unet_model(src_img)
            loss = criterion(output, tgt_img)
            #backword
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #densenet预测tgt_img
            densenet_tgt_img_out = densenet_model(tgt_img)
            _, tgt_img_predicted = torch.max(densenet_tgt_img_out.data, 1)
            

            #densenet预测src_trans_img
            densenet_src_trans_img_out = densenet_model(output)
            _, src_trans_img_predicted = torch.max(densenet_src_trans_img_out.data, 1)

            #展示参数
            #此次iter的参数
            iter_img_num = src_img.size(0)
            iter_tgt_img_correct = (tgt_img_predicted == label).sum().item()
            iter_src_trans_img_correct = (src_trans_img_predicted == label).sum().item()
            iter_loss = loss
            #一次epoch的总参数
            total += iter_img_num
            tgt_img_correct += iter_tgt_img_correct
            src_trans_img_correct += iter_src_trans_img_correct
            total_loss += iter_loss * iter_img_num

            #控制台打印参数
            if (i+1) % config.show_n_iter == 0:
                print('Epoch: [{}/{}], Step: [{}/{}], Loss: {:.6f}, tgt_img_correct_acc: {:.4f}, src_trans_img_correct_acc:{:.4f}'
                    .format(epoch+1, config.epoches, i+1, total_step, iter_loss, iter_tgt_img_correct/iter_img_num, iter_src_trans_img_correct/iter_img_num))
        tgt_img_correct_acc = tgt_img_correct/total
        src_trans_img_correct_acc = src_trans_img_correct/total
        avg_loss = total_loss/total
        if src_trans_img_max_acc<src_trans_img_correct_acc:
            src_trans_img_max_acc = src_trans_img_correct_acc
            print("New max acc: %.4f" % src_trans_img_max_acc)
            torch.save(unet_model.state_dict(), './result/unet_model.dat')
        with open(result_file_path, 'a') as f:
            f.write('%03d,%.6f,%.4f,%.4f\n' %(epoch, avg_loss, tgt_img_correct_acc, src_trans_img_correct_acc))
        

def load_data(config):
    mean = [0.5,]
    stdv = [0.2,]
    src_transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    tgt_transform = transforms.Compose([
        transforms.Resize((640, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    data_set = CustomDataset(filename="../data/deepbc/valid_autoencoder_labels/data.txt", src_dir="../data/deepbc/usg_images_cutted_p1", tgt_dir="../data/deepbc/usg_images_cutted_v3", src_transform=src_transform, tgt_transform=tgt_transform)

    #载入训练数据
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=config.batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))
    return data_loader

if __name__ == "__main__":
    config = TrainingConfig
    data_loader = load_data(config)
    #待训练unet模型
    unet_model = UNet(3, 3, bilinear=True)
    unet_model = unet_model.to(device)
    #载入训练好的densenet模型
    densenet_model = torch.load("./densenet_model/densenet_model.pkl")
    densenet_model = densenet_model.to(device)
    if config.train:
        train_valid(unet_model, densenet_model, config, data_loader)