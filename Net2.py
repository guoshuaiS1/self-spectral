import torch
import torch.nn as nn
import scipy.io as sio
from sklearn.cluster import KMeans
import evaluation
import numpy as np
import math
import torch.nn.functional as F
from losses import SupConLoss

best_paras=[]
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
        #self.S.copy_(torch.Tensor(Load_intial_S) #初始化图
        #self.S = Load_intial_S()
        #self.register_parameter("Spectral_embedding",self.F)
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
        labels=Load_labels()
        nmi, ari, f, acc = evaluation.evaluate(labels, label_pred)
        return acc,nmi,ari


def Load_intial_S():
    data_0 = sio.loadmat('./HW1256graph.mat')
    data_dict = dict(data_0)
    S_init = data_dict['S'][0, 0]  # 图
    labels = data_dict['gt']  # 2000
    labels = torch.from_numpy(labels).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
    S_init = torch.from_numpy(S_init).to(torch.float32)  # 转化为张量
    return S_init

def Load_labels():
    data_0 = sio.loadmat('./HW1256graph.mat')
    data_dict = dict(data_0)
    labels = data_dict['gt']  # 2000
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

class Constrint_loss(nn.Module): #将有约束问题转化为无约束问题
    def __init__(self):
        super(Constrint_loss, self).__init__()
    def forward(self,F):
        a=torch.mm(F.t(),F)
        I=torch.eye(F.size(1))
        loss=torch.pow(torch.norm(a-I),2)
        return loss

def initial_F0(): #2000*10 保证F0'*F0=I
    F0 = np.zeros((2000, 10))
    for i in range(10):
        F0[(200 * i):(200 * i + 200), i] = 1
    F0 = F0 * np.sqrt(1 / 200)
    np.random.shuffle(F0)
    # o=np.matmul((F0.T),F0)
    # print(o)
    return F0

mask=generate_mask(k=50) #k值参数可调
r=0.0001

lam1=1
lam2=0.0001
print('paras is :{},{}'.format(lam1,lam2))
spectral_net=Net(2500,5)
acc_log = []
nmi_log = []
optimizer = torch.optim.Adam(spectral_net.parameters(), lr=0.003)
T=500
loss1_criterion=spectral_loss()
loss2_criterion=Constrint_loss()
criterion = SupConLoss()
for k in range(1,T+1):
    spectral_net.train()
    F = spectral_net.forward()
    feature = torch.cat([F.unsqueeze(1), F.unsqueeze(1)], dim=1)
    optimizer.zero_grad()
    loss2 = criterion(feature, None, mask)
    loss1 = loss1_criterion(F,Load_intial_S())
    loss3 = loss2_criterion(F)
    loss = 5*loss1+lam1*loss2+ lam2 * loss3  # 可调参数
    loss.backward()
    optimizer.step()

    print('====> Epoch: {} spectral loss: {} contrastive loss: {}constraint loss: {} total loss:{}'.format(
        k, loss1.item(),loss2.item(),loss3.item(),loss.item()))

    spectral_net.eval()
    acc, nmi, _ = spectral_net.forward_cluster()
    acc_log.append(acc)
    nmi_log.append(nmi)
    print('====> Epoch: {} acc: {:.4f} nmi: {}'.format(
        T, acc, nmi))
print('max acc in {} is {}'.format(k,max(acc_log)))
sio.savemat('acc_'+str(lam1)+'_'+str(lam2)+'.mat',{'acc':acc_log})
sio.savemat('nmi_'+str(lam1)+'_'+str(lam2)+'.mat',{'nmi':nmi_log})
if(max(acc_log)>0.53):
    best_paras.append([lam1,lam2])
    print('best paras is lam1:{},lam2:{}'.format(lam1,lam2)
    )


sio.savemat('best_paras.mat',{'paras':best_paras})