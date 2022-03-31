
#down-top-down
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import resnet
from torchvision import datasets, transforms,models
import numpy as np
from torch.autograd import Variable
from nni.algorithms.compression.pytorch.pruning import LevelPruner
from blocktrim_res18 import Mask,PrunedModel,Strategy
import resnetn
import torch.nn.functional as F
from mb2 import MobileNetV2
import torch
from thop import profile

def l1_regularization(model, l1_alpha):
    l1_loss = []
    for module in model.modules():
        if type(module) is nn.BatchNorm2d:
            l1_loss.append(torch.abs(module.weight).sum())
    return l1_alpha * sum(l1_loss)

def train(epochs,net,lr):
    net.train()
    start_time = time.time()
    time_p, tr_acc, ts_acc, loss_p = [], [], [], []
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
    {'params': [p for n, p in net.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 1e-4},
    {'params': [p for n, p in net.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=lr, momentum=0.9)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)])
    testset = torchvision.datasets.CIFAR100(root = args.data, train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)

    criterion = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root = args.data, train=True, 
        transform=transforms.Compose([
                transforms.Pad(4, padding_mode='reflect'),
                transforms.RandomCrop(32, padding=4),
                #transforms.RandomResizedCrop(32, scale=(0.08, 1.0), ratio=(0.75, 1.33), interpolation=2),
                transforms.RandomHorizontalFlip(),
                #transforms.RandomRotation(30),
                transforms.ToTensor(),
                transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)])),
        batch_size = 128, shuffle=True, num_workers=4)
    for epoch in range(0,epochs):
        adjust_learning_rate_diss(optimizer, epoch)
        sum_loss, sum_acc, sum_step = 0., 0., 0.
        for batch_idx, (data, target) in enumerate(trainloader):       
            data, target = Variable(data.cuda()), Variable(target.cuda())
            output = net(data)
            
            smoothing = 0.1
            logprobs = torch.nn.functional.log_softmax(output, dim=-1)
            nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
            nll_loss = nll_loss.squeeze(1)
            smooth_loss = -logprobs.mean(dim=-1)
            loss = (1 - smoothing) * nll_loss + smoothing * smooth_loss
            loss = loss.mean()
            
            loss1 = criterion(output,target)
            sum_loss += loss1.item()*len(target)
            pred = torch.max(output, 1)[1]
            sum_acc += sum(pred==target).item()
            sum_step += target.size(0)
            optimizer.zero_grad()
            loss.backward()
        
            #***********************ϡ��ѵ������BN��ý���Լ����**************************
            #if args.sr:
                #updateBN()

            optimizer.step()
            best = -1
            if batch_idx % 100 == 0:
                for data, target in testloader:
                    data, target = Variable(data.cuda()), Variable(target.cuda())

                test_output = net(data)
                pred = torch.max(test_output, 1)[1]
                rightnum = sum(pred == target).item()
                acc = rightnum / float(target.size(0))
                
                print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".
                      format(epoch + 1, epochs,
                                    sum_loss/(sum_step), sum_acc/(sum_step), acc, time.time()-start_time))
                
                time_p.append(time.time() - start_time)
                tr_acc.append(sum_acc / sum_step)
                ts_acc.append(acc)
                loss_p.append(sum_loss / sum_step)
                acc = test(net)
                
                if best < acc:
                  best = acc
                  temp = net
    return net

def test(model):
    # ���ز�������
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root = args.data, train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
            np.array([63.0, 62.1, 66.7]) / 255.0)])),
        batch_size = 128, shuffle=False, num_workers=4)
    #model.eval()
    correct = 0

    for data, target in test_loader:
        data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)
    print('Accuracy: {:.2f}%\n'.format(acc))
    return acc



def diss(epochs,net_tea,net_stu,T,pre=None,momentum=0.9,heritate=False,l1_alpha=0.0):
    net_stu.train()
    start_time = time.time()
    time_p, tr_acc, ts_acc, loss_p = [], [], [], []
    optimizer = torch.optim.SGD(net_stu.parameters(), lr=0.001,momentum=0.9 ,weight_decay=5e-4)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    testset = torchvision.datasets.CIFAR100(root = args.data, train = False, download = True, transform = transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=0)

    criterion_soft = nn.KLDivLoss()
    criterion_hard = nn.CrossEntropyLoss()
    trainloader = torch.utils.data.DataLoader(
        datasets.CIFAR100(root = args.data, train=True, transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size = 64, shuffle=True, num_workers=0)
    best = -1
    for epoch in range(0,epochs):
        test(net_stu)
        #torch.save(model_ta.state_dict(),"ta2stu-cifar10.pth")
        #adjust_learning_rate(optimizer, epoch)
        #train(epoch)
        #test()
        sum_loss, sum_acc, sum_step = 0., 0., 0.
        for batch_idx, (data, target) in enumerate(trainloader):      
             
            data, target = Variable(data.cuda()), Variable(target.cuda())
            #print(data.size())
            # data shape: [50, 3, 32, 32]
            #print(target.size())
            # target shape: [50]
            output_hard = net_stu(data)
            output_soft = net_tea(data)
            loss_hard = criterion_hard(output_hard, target)
            loss_soft = criterion_soft(F.log_softmax(output_hard/T,dim=1), F.softmax(output_soft/T,dim=1))
            if heritate:
                output_pre = pre(data)
                loss_pre = criterion_soft(F.log_softmax(output_hard/T,dim=1), F.softmax(output_pre/T,dim=1))
                loss = 0.9 * (1-momentum) * loss_soft*T*T +0.9 *momentum *loss_pre+ 0.1 * loss_hard+l1_regularization(pre,l1_alpha)
                
            else:
                loss = 0.9 * loss_soft*T*T +0.1 * loss_hard
            #pred = output.data.max(1, keepdim=True)[1]
            sum_loss += loss.item()*len(target)
            pred = torch.max(output_hard, 1)[1]
            sum_acc += sum(pred==target).item()
            sum_step += target.size(0)
            optimizer.zero_grad()
            loss.backward()
        
            #***********************ϡ��ѵ������BN��ý���Լ����**************************
            #if args.sr:
                #updateBN()

            optimizer.step()
            #test(net_stu)

            if batch_idx % 100 == 0:
                for data, target in testloader:
                    data, target = Variable(data.cuda()), Variable(target.cuda())

                test_output = net_stu(data)
                pred = torch.max(test_output, 1)[1]
                rightnum = sum(pred == target).item()
                acc = rightnum / float(target.size(0))
                if acc > best:
                  best = best
                  ##to be determined
                print("epoch: [{}/{}] | Loss: {:.4f} | TR_acc: {:.4f} | TS_acc: {:.4f} | Time: {:.1f}".
                      format(epoch + 1, epochs,
                                    sum_loss/(sum_step), sum_acc/(sum_step), acc, time.time()-start_time))

                time_p.append(time.time() - start_time)
                tr_acc.append(sum_acc / sum_step)
                ts_acc.append(acc)
                loss_p.append(sum_loss / sum_step)
    return net_stu
    

def adjust_learning_rate_diss(optimizer, epoch):
    #warm_list = [20, 30, 40]
    update_list = [100, 150]
    #update_list = [30, 60, 90, 120, 150, 180]
    '''
    if epoch in warm_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 10
    '''
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return


parser = argparse.ArgumentParser()

parser.add_argument('--data', action='store', default='data.cifar100',help='dataset path')

args = parser.parse_args()


model = resnetn.cifarresnet18(num_classes=100).cuda()
teacher = resnetn.cifarresnet18(num_classes=100).cuda()
model.load_state_dict(torch.load("res18-100-7527.pth"))
teacher.load_state_dict(torch.load("res18-100-7527.pth"))
pre=teacher
lr=0.1

for i in range(3):
  train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.1,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.1.pth")
pre = model

lr = 0.1
for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.2,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.2.pth")
pre = model

lr = 0.1

for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.3,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
lr = 0.1
torch.save(model.state_dict(),"pru-res18-10-0.3.pth")
pre = model

for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  model = (model+pre)/(2*sqrt(abs(model-pre)))
  mask = Strategy.prune(0.4,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.4.pth")
lr = 0.1

for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.5,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.5.pth")

lr = 0.1
for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.6,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
#lr /= 10
torch.save(model.state_dict(),"pru-res18-10-0.6.pth")

lr = 0.1
for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.7,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.7.pth")

lr = 0.1
for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.8,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.8.pth")
lr = 0.1
for i in range(3):
  model = train(10,model,lr)
  for name,para in model.named_parameters():
    for name1,para1 in pre.named_parameters():
      if name==name1:
        para = (para+para1)/(2*torch.sqrt(abs(para-para1)))
  mask = Strategy.prune(0.9,model,block=i+1)
  prunedmodel = PrunedModel(model.cpu(),mask,i+1).cuda()
  print("After step"+str(i)+'.'+':',test(prunedmodel)) 
  model = diss(10,teacher,model,200,pre,0.9,True)
  if (i == 1) or (i == 2 ):lr /= 10
      
torch.save(model.state_dict(),"pru-res18-10-0.9.pth")


