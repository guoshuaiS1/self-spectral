import torch
import torch.nn as nn
import scipy.io as sio
from sklearn.cluster import KMeans
import evaluation
import numpy as np
import math
import torch.nn.functional as F
from losses import SupConLoss
acc_log=[]
nmi_log=[]
acc1_log=[]
nmi1_log=[]
criterion=SupConLoss()

class Net(nn.Module):
    def __init__(self,N,k):
        #N表示样本数量
        #k表示类别数量
        super(Net,self).__init__()
        self.N=N
        self.k=k
        #F = torch.randn((N,k),requires_grad=True)
        #self.F = torch.nn.Parameter(torch.Tensor(self.N,self.k),requires_grad=True) #谱嵌入
        self.F = torch.nn.Parameter(torch.tensor(initial_F0()), requires_grad=True)
        #self.S = torch.nn.Parameter(torch.Tensor(N,N),requires_grad=False) #图
        #self.S.copy_(torch.Tensor(Load_intial_S()))
        #self.S = Load_intial_S()
        self.register_parameter("Spectral_embedding",self.F)
        #self.reset_parameters()

    '''def reset_parameters(self):
        stdv = 0.01
        self.F.data.uniform_(-stdv, stdv)'''


    def forward(self):
        return self.F

    def forward_cluster(self):
        estimator = KMeans(n_clusters=self.k, n_init=20)  # 聚类数量
        F=self.F.detach()
        F=F.cpu().detach().numpy()
        estimator.fit(F)
        label_pred = estimator.labels_

        return label_pred

    def update_Y(self):
        Y = torch.zeros_like(self.F).type(torch.long)

        index_max = torch.argmax(self.F, dim=1, keepdim=False)
        for i in range(len(index_max)):
            Y[i, index_max[i]] = 1.
        return Y


def Load_intial_S():
    data_0 = sio.loadmat('./.aff.mat')
    data_dict = dict(data_0)
    S_init = data_dict['S']  # 图
    S_init = torch.from_numpy(S_init).to(torch.float32)  # 转化为张量
    return S_init

def Load_labels():
    data_0 = sio.loadmat('./bdgp_label.mat')
    data_dict = dict(data_0)
    labels = data_dict['Ytr']  # 2000
    labels = torch.from_numpy(labels).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
    labels = np.array(labels)
    return labels

def generate_mask(k=20): #生成正负样本关系矩阵，正样本之间是1，负样本之间是0

    S_init=Load_intial_S()
    S=torch.diag_embed(torch.diag(S_init))
    S=np.array((S_init-S))
    mask = np.zeros_like(S)
    sort_index=np.argsort(S,axis=1)
    n=sort_index.shape[1]
    sort_index_topk=sort_index[:,(n-k):n]
    for i in range(len(sort_index_topk)):
        for j in sort_index_topk[i,:]:
            mask[i,j]=1
    mask=mask+np.eye(n)
    o=torch.tensor(mask-mask.T)
    #print(o.size())
    #print(o)
    if(torch.norm(o)==0):
        print('mask is 对称的')
    else:
        print('mask is not 对称的')
    return torch.tensor(mask).float()

def compute_Y(F):
    Y = torch.zeros_like(F).type(torch.long)

    index_max = torch.argmax(F, dim=1, keepdim=False)
    for i in range(len(index_max)):
        Y[i, index_max[i]] = 1.
    return Y
def generate_prolabels(Y): #根据Y（n*k)产生伪标签
    label = torch.topk(Y, 1)[1].squeeze(1)
    return label


'''def generate_mask(rol=0.0001): #生成正负样本关系矩阵，正样本之间是1，负样本之间是0
    #rol为设置的判断阈值
    S_init=Load_intial_S()
    S=torch.diag_embed(torch.diag(S_init))
    S=np.array((S_init-S))
    mask = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if(S[i,j]>rol):
                mask[i,j]=1
    return torch.tensor(mask).float()'''

class Self_paced_loss(nn.Module):
    def __init__(self,select=False):
        super(Self_paced_loss, self).__init__()
        self.select=select

    def compute_discriminator(self,F,S):
        distance_mat = torch.norm(F[:, None] - F, dim=2, p=2)
        distance_mat = torch.pow(distance_mat,2)
        losses=torch.zeros((F.size(0)))
        for i in range(F.size(0)):
            d=distance_mat[i,:]
            s_ij=S[i,:]
            losses[i]=torch.sum(d*s_ij)

        return losses

    def compute_sample_weight(self, discr, t, T):
        lam = torch.mean(discr) + t * torch.std(discr) / T
        sample_weight = torch.where(discr < lam, 1., 0.1)
        select_num = torch.sum(sample_weight).int()
        return select_num, sample_weight

    def forward(self, F, S,t,T):
        losses=self.compute_discriminator(F,S)
        if self.select==False:
            loss=torch.sum(losses)
        else:
            num,sample_weight=self.compute_sample_weight(losses,t,T)
            loss=torch.sum(losses*sample_weight)
        return loss

class Constrint_loss(nn.Module): #将有约束问题转化为无约束问题
    def __init__(self):
        super(Constrint_loss, self).__init__()
    def forward(self,F):
        a=torch.mm(F.t(),F)
        I=torch.eye(F.size(1))
        loss=torch.pow(torch.norm(a-I),2)
        return loss

class spectral_loss(nn.Module):
    def __init__(self):
        super(spectral_loss, self).__init__()
    def forward(self,F,S):
        S=S.to(torch.float32)
        F=F.to(torch.float32)
        D=torch.sum(S,dim=1)
        D=torch.diag_embed(D)
        L=D-S
        loss=torch.trace(torch.matmul(torch.matmul(F.T,L),F))
        return loss

class discrete_loss(nn.Module):
    def __init__(self):
        super(discrete_loss, self).__init__()
    def forward(self,F,Y):
        F=F.to(torch.float32)
        Y=Y.to(torch.float32)
        loss=torch.trace(torch.matmul(Y,F.T))
        return -loss

def update_Y(F): #F:n*c
    Y=torch.zeros_like(F).type(torch.long)

    index_min=torch.max(Y,dim=1,keepdim=False)
    for i in range(len(index_min)):
        Y[i,index_min[i]]=1.
    return torch.tensor(Y)

def clustering_result(Y):
    n=Y.size(0)
    c=Y.size(1)
    Y=np.array(Y)
    preds=np.zeros(n)
    for i in range(n):
        for j in range(c):
            if(Y[i,j]!=0):
                preds[i]=j
    labels = Load_labels()
    preds=preds.astype(np.int32)
    nmi, ari, f, acc = evaluation.evaluate(labels, preds)
    return acc,nmi,ari

def initial_F0(): #2000*10 保证F0'*F0=I
    F0 = np.zeros((2500, 5))
    for i in range(5):
        F0[(500 * i):(500 * i + 500), i] = 1
    F0 = F0 * np.sqrt(1 / 500)
    np.random.shuffle(F0)
    # o=np.matmul((F0.T),F0)
    # print(o)
    return F0
spectral_net=Net(2500,5)

optimizer = torch.optim.Adam(spectral_net.parameters(), lr=0.003)
T=800
loss1_criterion=spectral_loss()
loss2_criterion=Constrint_loss()
loss3_criterion=discrete_loss()
mask=generate_mask(k=50) #k值参数可调
r=0.001

def optimize(epoch):
    import torch.nn.functional as F

    F0=initial_F0()
    o=1  #F-F0
    k=0  #记录迭代轮数，每轮有epoch次深度优化
    while(o>r):
        k+=1
        F0=torch.tensor(F0)
        for i in range(epoch):
            spectral_net.train()
            F=spectral_net.forward()
            feature=torch.cat([F.unsqueeze(1), F0.unsqueeze(1)], dim=1)
            optimizer.zero_grad()
            loss1 = criterion(feature,None,mask)
            loss2 = loss2_criterion(F)
            loss = loss1 + 10*loss2   #可调参数
            loss.backward()
            optimizer.step()

            print('====> Epoch: {}  contrastive loss: {:.4f} constraint loss: {}'.format(
                k, loss1.item(), loss2.item()))
        spectral_net.eval()
        global acc_log
        global nmi_log
        acc, nmi, _ = spectral_net.forward_cluster()
        acc_log.append(acc)
        nmi_log.append(nmi)
        print('====> Epoch: {} acc: {:.4f} nmi: {}'.format(
            k, acc, nmi))
        F=spectral_net.forward()
        F=F.cpu().detach()
        F0=F0.numpy()
        o = np.linalg.norm(F-F0,np.inf)  #求无穷范数

        print("o:",o)
        F0=F


def train(epoch,label_pred):
    spectral_net.train()
    if epoch<1:
        F = spectral_net.forward()
        feature = torch.cat([F.unsqueeze(1), F.unsqueeze(1)], dim=1)
        optimizer.zero_grad()
        loss2 = criterion(feature, None, mask)
        loss1 = loss1_criterion(F, Load_intial_S())
        loss3 = loss2_criterion(F)
        loss = loss1 + 0.01 * loss2 + 0.01 * loss3  # 可调参数
        loss.backward()
        optimizer.step()

        print(
            '====> Epoch: {}  spetral loss: {} contrastive loss: {}constraint loss: {}total loss:{}'.format(
                epoch, loss1.item(), loss2.item(), loss3.item(), loss.item()))

    if ((epoch>=1)and(epoch<=240)):
        F = spectral_net.forward()
        Y = spectral_net.update_Y()
        feature = torch.cat([F.unsqueeze(1), F.unsqueeze(1)], dim=1)
        optimizer.zero_grad()
        loss2 = criterion(feature, None, mask)
        loss1 = loss1_criterion(F,Load_intial_S())
        loss3 = loss2_criterion(F)
        loss4 = loss3_criterion(F,Y)
        loss=0.00001*loss1 + 1*loss2+ 0.001 * loss3 + 0.0001*loss4 # 可调参数
        loss.backward()
        optimizer.step()

        print('====> Epoch: {}  spetral loss: {} contrastive loss: {}constraint loss: {}discrete loss:{} total loss:{}'.format(
            epoch, loss1.item(),loss2.item(),loss3.item(),loss4.item(),loss.item()))
    if ((epoch >240)):
        F = spectral_net.forward()
        Y = spectral_net.update_Y()
        feature = torch.cat([F.unsqueeze(1), F.unsqueeze(1)], dim=1)
        optimizer.zero_grad()
        '''F = F.cpu().detach().numpy()
        estimator = KMeans(n_clusters=10, n_init=20).fit(F)
        prolabels = torch.from_numpy(estimator.labels_)'''
        prolabels=torch.from_numpy(label_pred)
        #prolabels = label_pred
        loss2 = criterion(feature, prolabels, None)
        loss1 = loss1_criterion(F, Load_intial_S())
        loss3 = loss2_criterion(F)
        loss4 = loss3_criterion(F, Y)
        loss=0.00001*loss1 + 1*loss2+ 0.001 * loss3 + 0.0001*loss4 # 可调参数
        loss.backward()
        optimizer.step()

        print(
            '====> Epoch: {}  spetral loss: {} contrastive loss: {}constraint loss: {}discrete loss:{} total loss:{}'.format(
                epoch, loss1.item(), loss2.item(), loss3.item(), loss4.item(), loss.item()))

def test(epoch):
    spectral_net.eval()
    label_pred=spectral_net.forward_cluster()
    '''acc,nmi,_=spectral_net.forward_cluster()
    F = spectral_net.forward()
    Y = compute_Y(F)
    acc1,nmi1,_=clustering_result(Y)
    acc_log.append(acc)
    nmi_log.append(nmi)
    acc1_log.append(acc1)
    nmi1_log.append(nmi1)

    print('====> kmeans: Epoch: {} acc: {:.4f} nmi: {}'.format(
          epoch, acc,nmi))'''
    F = spectral_net.forward()
    Y = compute_Y(F)
    acc1, nmi1, _ = clustering_result(Y)
    print('====> Y result: Epoch: {} acc: {:.4f} nmi: {}'.format(
        epoch, acc1, nmi1))
    label = torch.topk(Y, 1)[1].squeeze(1)
    return label_pred,label

label_pred1=np.array(np.zeros(2500))
labels = Load_labels()
counter=0
for i in range(T):
    train(i+1,label_pred1)
    label_pred,label1=test(i+1)

    nmi, ari, f, acc = evaluation.evaluate(labels, label_pred)

    print('====> kmeans: Epoch: {} acc: {:.4f} nmi: {}'.format(
        i+1, acc, nmi))
    acc_log.append(acc)
    nmi_log.append(nmi)
    if(counter%10==0):
        label_pred1=label_pred
    counter += 1
#torch.save(spectral_net.state_dict(), './models/spectral.pth')
#optimize(10)



print(max(acc_log))
index=acc_log.index(max(acc_log))
print(nmi_log[index])
print("********")
'''print(max(acc1_log))
index1=acc1_log.index(max(acc1_log))
print(nmi1_log[index1])'''
