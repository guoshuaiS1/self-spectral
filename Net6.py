import torch
import torch.nn as nn
import scipy.io as sio
from sklearn.cluster import KMeans
import evaluation
import numpy as np
import math
import torch.nn.functional as F
from losses import SupConLoss
import metrics
acc_log=[]
nmi_log=[]
acc1_log=[]
nmi1_log=[]

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
        self.Y_ = torch.nn.Parameter(torch.randn(self.N,self.k), requires_grad=True)
        self.func = torch.nn.Softmax(dim=1)
        #self.S = torch.nn.Parameter(torch.Tensor(N,N),requires_grad=False) #图
        #self.S.copy_(torch.Tensor(Load_intial_S()))
        #self.S = Load_intial_S()
        self.register_parameter("Spectral_embedding",self.F)
        self.register_parameter("soft label",self.Y_)
        #self.reset_parameters()

    '''def reset_parameters(self):
        stdv = 0.01
        self.F.data.uniform_(-stdv, stdv)'''


    def forward(self):
        Y=self.func(self.Y_)
        F=self.F
        return F,Y

    def forward_cluster(self):
        estimator = KMeans(n_clusters=self.k, n_init=20,random_state=10)  # 聚类数量
        F = self.F.detach()
        F = F.cpu().detach().numpy()
        estimator.fit(F)
        label_pred = estimator.labels_
        labels = Load_labels()
        nmi, ari, f, acc = evaluation.evaluate(labels, label_pred)
        return acc, nmi, ari


def Load_intial_S():
    data_0 = sio.loadmat('./.aff.mat')
    data_dict = dict(data_0)
    S_init = data_dict['S']# 图
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
#spectral_net.load_state_dict(torch.load('./models/spectral_bdgp241.pth'))
optimizer = torch.optim.Adam(spectral_net.parameters(), lr=0.003)
T=500
loss1_criterion=spectral_loss()
loss2_criterion=Constrint_loss()
criterion=SupConLoss()
labels=Load_labels()
r=0.001

def test(epoch):
    spectral_net.eval()
    acc, nmi, _ = spectral_net.forward_cluster()
    print('====> Epoch: {} acc: {:.4f} nmi: {}'.format(
        epoch, acc, nmi))

def train(epoch):
    spectral_net.train()
    F,Y = spectral_net.forward()
    Y = Y.to(torch.float32)
    F = F.to(torch.float32)
    F_Y = torch.mm(Y,torch.inverse(torch.mm(Y.T,Y)))

    F_Y = torch.mm(torch.mm(F_Y,Y.T), F)
    feature = torch.cat([torch.nn.functional.normalize(F,p=2,dim=1).unsqueeze(1), torch.nn.functional.normalize(F_Y,p=2,dim=1).unsqueeze(1)], dim=1)
    label_pred = generate_prolabels(Y)
    optimizer.zero_grad()
    loss2 = criterion(feature, label_pred)  #对比损失
    loss1 = loss1_criterion(F, Load_intial_S()) #普损失
    loss3 = loss2_criterion(F)  #约束损失
    loss = loss1 + 0 * loss2 + 1 * loss3  # 可调参数
    loss.backward()
    optimizer.step()
    nmi, ari, f, acc = evaluation.evaluate(labels, label_pred.numpy())
    #acc = metrics.acc(labels,label_pred.numpy())
    #nmi = metrics.nmi(labels,label_pred.numpy())
    print('====> Epoch: {}  spetral loss: {} contrastive loss: {}constraint loss: {} total loss:{}'.format(
        epoch, loss1.item(), loss2.item(), loss3.item(), loss.item()))
    print('====> Epoch: {}  Y label acc: {} Y label nmi: {}'.format(
        epoch, acc,nmi))
    global acc_log
    global nmi_log
    acc_log.append(acc)
    nmi_log.append(nmi)

for i in range(T):
    train(i+1) #训练的同时采用离散标签Y做测试
    test(i+1)   #采用K_means对谱嵌入F做测试

print("********")
print('统计最优情况：')
print(max(acc_log))
index=acc_log.index(max(acc_log))
print(nmi_log[index])

'''print(max(acc1_log))
index1=acc1_log.index(max(acc1_log))
print(nmi1_log[index1])'''
