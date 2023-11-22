import scipy.io as sio
import torch
import numpy as np
from losses import SupConLoss
import torch.nn.functional as F
criterion=SupConLoss()
def Load_intial_S():
    data_0 = sio.loadmat('./HW1256graph.mat')
    data_dict = dict(data_0)
    S_init = data_dict['S'][0,0]  # 图
    labels = data_dict['gt']  # 2000
    labels = torch.from_numpy(labels).to(torch.int32).view(-1)  # 转化为张量 .view()可改变尺寸
    S_init = torch.from_numpy(S_init).to(torch.float32)  # 转化为张量
    return S_init

def generate_mask(k=20):

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
    print(o.size())
    print(o)
    if(torch.norm(o)==0):
        print('mask is 对称的')
    else:
        print('mask is not 对称的')
    return torch.tensor(mask).float()
'''
mask=generate_mask(k=0)
mask=torch.eye(2000)
f1=torch.randn((2000,10))
f2=torch.randn((2000,10))
f1=F.normalize(f1,p=2,dim=1)
f2=F.normalize(f2,p=2,dim=1)
features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
print(features.size())
loss=criterion(features,None,mask)
print(loss)'''
def initial_F0(): #2000*10 保证F0'*F0=I
    F0=np.zeros((2000,10))
    for i in range(10):
        F0[(200*i):(200*i+200),i]=1
    F0=F0*np.sqrt(1/200)
    np.random.shuffle(F0)
    print(F0)
    o=np.matmul((F0.T),F0)
    print(o)
    return F0

def generate_mask(rol=0.0001): #生成正负样本关系矩阵，正样本之间是1，负样本之间是0
    #rol为设置的判断阈值
    S_init=Load_intial_S()
    S=torch.diag_embed(torch.diag(S_init))
    S=np.array((S_init-S))
    mask = np.zeros_like(S)
    for i in range(S.shape[0]):
        for j in range(S.shape[1]):
            if(S[i,j]>rol):
                mask[i,j]=1
    return torch.tensor(mask).float()
