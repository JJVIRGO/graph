import torch
from torch import nn, optim
import os
from tifdataset import TIFDataset
from Lenet52D import VGG
from torch.utils.data import DataLoader
import vgg
from DenseNet121 import DenseNet121
from scipy.stats import pearsonr
os.chdir("f:/10x/Xenium_V1_FFPE_TgCRND8_17_9_months_outs/")
def initialize_weights_kaiming(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            nn.init.constant_(m.bias, 0)

batchsz = 64
lr = 0.0001
epochs = 200
device = torch.device('cuda')

train_db = TIFDataset(img_root='total', csv_root='transtotal', mode='train')
val_db = TIFDataset(img_root='total', csv_root='transtotal', mode='val')
test_db = TIFDataset(img_root='total', csv_root='transtotal', mode='test')
train_loader = DataLoader(train_db, batch_size=batchsz, shuffle=True, num_workers=4)
val_loader = DataLoader(val_db, batch_size=batchsz, num_workers=2)
test_loader = DataLoader(test_db, batch_size=batchsz, num_workers=2)

def rounded_loss(logits, targets, alpha=0.1):
    # Standard MSE loss
    mse_loss = nn.MSELoss()(logits, targets)
    # Difference between logits and their rounded values
    #round_diff = nn.MSELoss()(log     its, logits.round())
    # Combine the losses
    #total_loss = mse_loss + alpha * round_diff
    total_loss=mse_loss
    return total_loss

def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def RMSELoss(logits, targets):
    mse_loss = nn.MSELoss()(logits, targets)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss
def main1():
    model = DenseNet121(347).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-2)
    lambda_l1 = 0.001  # L1 正则化项的权重        
    
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # 使用RMSE损失函数
            loss = RMSELoss(logits, y)

            # 添加L1正则化
            l1_loss = l1_regularization(model, lambda_l1)
            total_loss = loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print(epoch, step, 'loss:', loss.item())
            # if step % 1000 == 0:
            #     torch.save(model.state_dict(), 'best.mdl')
        model.eval()
        total_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x) 
                loss = rounded_loss(logits, y, alpha=0.1)
                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        print(epoch, 'val loss:', avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(model.state_dict(), 'best.mdl')
            print(f"Saved best model in epoch {epoch} with val loss: {avg_loss}")
        print('Best val loss:', best_loss, 'in epoch:', best_epoch)
    
    # Load the best model
    predictions = []
    actuals = []
    model.load_state_dict(torch.load('best.mdl'))
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = rounded_loss(logits, y, alpha=0.1)
            total_loss += loss.item()
            predictions.extend(logits.view(-1).cpu().numpy())
            actuals.extend(y.view(-1).cpu().numpy())

    avg_loss = total_loss / len(test_loader)    
    print('Test loss:', avg_loss)
    # 计算皮尔逊相关系数
    pcc, _ = pearsonr(predictions, actuals)
    print(f"Pearson Correlation Coefficient: {pcc}")
    # Show a single prediction
    x, y = next(iter(test_loader))
    x = x.to(device)
    with torch.no_grad():
        prediction = model(x)
    for i in range(0,5):
        print('A single prediction:', prediction[i])
        print(y[i])

def main():
    x, y = next(iter(train_loader))
    x = x.to(device)
    print(x.shape,y.shape)
if __name__ == '__main__':
    main1()

'''
def l1_regularization(model, lambda_l1):
    l1_loss = 0
    for param in model.parameters():
        l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def RMSELoss(logits, targets):
    mse_loss = nn.MSELoss()(logits, targets)
    rmse_loss = torch.sqrt(mse_loss)
    return rmse_loss

def main1():
    model = DenseNet121(347).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
    lambda_l1 = 0.001  # L1 正则化项的权重
    best_loss = float('inf')
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            logits = model(x)

            # 使用RMSE损失函数
            loss = RMSELoss(logits, y)

            # 添加L1正则化
            l1_loss = l1_regularization(model, lambda_l1)
            total_loss = loss + l1_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()'''