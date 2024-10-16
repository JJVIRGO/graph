# import torch
# import torch.nn as nn
# import torchvision.models as models
# import torchvision.transforms as transforms
# import tifffile
# from torch.nn import functional as F
# class DenseNet121(nn.Module):
#     def __init__(self, num_classes):
#         super(DenseNet121, self).__init__()
#         # 加载预训练的DenseNet-121模型
#         densenet = models.densenet121(pretrained=True)

#         # 修改第一个卷积层以接受单通道图像
#         self.features = densenet.features
#         self.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

#         # 保留DenseNet的其他部分不变
#         self.classifier = densenet.classifier

#         # 为基因表达预测定义AuxNet
#         self.gene_classifiers = nn.ModuleList([
#             nn.Linear(num_classes, 1) for _ in range(num_classes)
#         ])
#     def forward(self, x):
#         features = self.features(x)
#         out = F.relu(features, inplace=True)
#         out = F.adaptive_avg_pool2d(out, (1, 1))
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#         return out

# def main():
#     num_classes = 347  # 假设有347个类别（或基因表达）
#     model = DenseNet121(num_classes)

#     image=torch.rand(1,1,224,224)
#     output = model(image)
#     print(output.shape)

# if __name__ == '__main__':
#     main()
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

class DenseNet121(nn.Module):
    def __init__(self,num_genes):
        super(DenseNet121, self).__init__()
        # 加载预训练的DenseNet-121模型
        densenet = models.densenet121(pretrained=True)

        # 修改第一个卷积层以接受单通道图像
        self.features = densenet.features
        self.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # 保留DenseNet的特征提取部分不变
        num_features = densenet.classifier.in_features

        # 为基因表达预测定义AuxNet
        self.gene_classifiers = nn.ModuleList([
            nn.Linear(num_features, 1) for _ in range(num_genes)
        ])

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # 为每个基因生成一个预测
        gene_outputs = [classifier(out) for classifier in self.gene_classifiers]
        return torch.cat(gene_outputs, dim=1)

def main():
    num_genes = 347 
    model = DenseNet121(num_genes)

    image = torch.rand(1, 1, 224, 224)  # 假设输入图像大小为224x224
    output = model(image)
    print(output.shape)  # 输出应该是[1, num_genes]

if __name__ == '__main__':
    main()

