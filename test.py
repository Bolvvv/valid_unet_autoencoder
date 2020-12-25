from os import name
import torch
from torch.utils import data
from torchvision import datasets, transforms, models
from dataset import CustomDataset
from config import TrainingConfig
from unet import UNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config  = TrainingConfig

def load_data():
    mean = [0.5,]
    stdv = [0.2,]
    src_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])
    tgt_transform = transforms.Compose([
        transforms.Resize((320, 240)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    label_path = "../data/deepbc/labels_1218/"
    image_path = "../data/deepbc/usg_images_cutted_p1"

    dataset_list = [label_path+"valid_BX.txt", label_path+"valid_CMGH.txt", label_path+"valid_Malignant_DeYang.txt"]
    data_set = CustomDataset(data_set_list=dataset_list, src_dir="../data/deepbc/usg_images_cutted_p1", tgt_dir="../data/deepbc/usg_images_cutted_v3", src_transform=src_transform, tgt_transform=tgt_transform)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=config.test_batch_size, shuffle=True,
                                              pin_memory=(torch.cuda.is_available()))
    return data_loader

def test(unet_model, densenet_model, data_loader):
    densenet_model.eval()
    unet_model.eval()
    with torch.no_grad():
        total = 0
        pre_photo_correct_num = 0
        pre_output_correct_num = 0
        pre_electronic_correct_num = 0
        for i, (src_img, tgt_img, label) in enumerate(data_loader):
            src_img = src_img.to(device)
            tgt_img = tgt_img.to(device)
            label = label.to(device)
            #生成图片
            output = unet_model(src_img)
            #预测照片
            pre_photo = densenet_model(src_img)
            _, pre_photo_result = torch.max(pre_photo.data, 1)
            #预测生成图片
            pre_output = densenet_model(output)
            _, pre_output_result = torch.max(pre_output.data, 1)
            #预测电子图片
            pre_electronic  = densenet_model(tgt_img)
            _, pre_electronic_result = torch.max(pre_electronic.data, 1)

            #计算参数
            total += src_img.size(0)
            pre_photo_correct_num += (pre_photo_result == label).sum().item()
            pre_output_correct_num += (pre_output_result == label).sum().item()
            pre_electronic_correct_num += (pre_electronic_result == label).sum().item()
        
        pre_photo_acc = pre_photo_correct_num/total
        pre_output_acc = pre_output_correct_num/total
        pre_electronic_acc = pre_electronic_correct_num/total

        print("预测照片准确率：%.4f，预测转换后图片准确率：%.4f，预测电子图片准确率：%.4f\n" % (pre_photo_acc, pre_output_acc, pre_electronic_acc))
        
if __name__ == "__main__":
    #载入数据
    data_loader = load_data()
    #载入Unet模型
    unet_model = UNet(3, 3, bilinear=True)
    unet_model = unet_model.to(device)
    unet_model.load_state_dict(torch.load('./result/unet_model.dat'))
    #载入densenet模型
    densenet_model = torch.load("./densenet_model/densenet_model.pkl")
    test(unet_model, densenet_model, data_loader)