import torch
from torch import nn,optim,autograd
import numpy as np
from torch.nn.modules.linear import Linear
import visdom
from torch.nn import functional as F
from matplotlib import pyplot as plt
import random

# 参数
h_dim=400
batch_size=512
viz=visdom.Visdom()

# G(z)
class Generator(nn.Module):

    def __init__(self):
        super(Generator,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,2),
        )

    def forward(self,z):
        output=self.net(z)
        return output

# D(x)
class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator,self).__init__()

        self.net=nn.Sequential(
            nn.Linear(2,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,h_dim),
            nn.ReLU(True),
            nn.Linear(h_dim,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        output=self.net(x)
        return output

# 生成数据
def data_generator():

    scale=2.
    centers=[
        (1, 0),
        (-1, 0),
        (0, 1),
        (0, -1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    centers=[(scale*x,scale*y) for x,y in centers]
    while True:
        dataset=[]
        for i in range(batch_size):
            point=np.random.randn(2)*.02
            center=random.choice(centers)
            point[0]+=center[0]
            point[1]+=center[1]
            dataset.append(point)
        dataset=np.array(dataset,dtype='float32')
        dataset/=1.414 # stdev
        yield dataset

# 绘制等高线图
def generate_image(D,G,xr,epoch):
    """
    Generates and saves a plot of the true distribution, the generator, and the
    critic.
    """
    N_POINTS=128
    RANGE=3
    plt.clf()

    points=np.zeros((N_POINTS,N_POINTS,2),dtype='float32')
    points[:,:,0]=np.linspace(-RANGE,RANGE,N_POINTS)[:,None]
    points[:,:,1]=np.linspace(-RANGE,RANGE,N_POINTS)[None,:]
    points=points.reshape((-1,2))

    # draw contour
    with torch.no_grad():
        points=torch.Tensor(points).cuda()
        disc_map=D(points).cpu().numpy() # D(x)
    x=y=np.linspace(-RANGE,RANGE,N_POINTS)
    cs=plt.contour(x,y,disc_map.reshape((len(x),len(y))).transpose()) # transpose():转置
    plt.clabel(cs,inline=1,fontsize=10) # 标注

    # draw samples
    with torch.no_grad():
        z=torch.randn(batch_size,2).cuda()
        samples=G(z).cpu().numpy()
    plt.scatter(xr[:, 0].cpu(), xr[:, 1].cpu(), c='orange', marker='.')
    plt.scatter(samples[:, 0], samples[:, 1], c='green', marker='+')

    viz.matplot(plt,win='contour',opts=dict(title='p(x):%d'%epoch))

def weights_init(m):
    if isinstance(m,nn.Linear):
        nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0)

# 惩罚项
def gradient_penalty(D,xr,xf):
    """

    :param D:
    :param xr:
    :param xf:
    :return:
    """
    LAMBDA=0.3
    xf=xf.detach()
    xr=xr.detach()

    alpha=torch.rand(batch_size,1).cuda()
    alpha=alpha.expand_as(xr)

    interpolates=alpha*xr+(1-alpha)*xf
    interpolates.requires_grad_()

    disc_interpolates=D(interpolates)

    gradients=autograd.grad(outputs=disc_interpolates,inputs=interpolates,
                            grad_outputs=torch.ones_like(disc_interpolates),
                            create_graph=True,retain_graph=True,only_inputs=True)[0]

    gp=((gradients.norm(2,dim=1)-1)**2).mean()*LAMBDA # torch.norm():求张量的范数

    return gp

def main():
    torch.manual_seed(23)
    np.random.seed(23)

    G=Generator().cuda()
    D=Discriminator().cuda()
    G.apply(weights_init)
    D.apply(weights_init)

    optim_G=optim.Adam(G.parameters(),lr=1e-3,betas=(0.5,0.9))
    optim_D=optim.Adam(D.parameters(),lr=1e-3,betas=(0.5,0.9))

    data_iter=data_generator() # 生成Dataset
    print('batch:',next(data_iter).shape)

    viz.line([[0,0]],[0],win='loss',opts=dict(title='loss',legend=['D','G'])) 

    for epoch in range(50000):

        # 1. train discriminator for k steps
        for _ in range(5):
            x=next(data_iter)
            xr=torch.from_numpy(x).cuda()

            predr=(D(xr))
            # max log(lossr)
            lossr=-(predr.mean())

            z=torch.randn(batch_size,2).cuda()
            # stop gradient on G
            xf=G(z).detach()

            predf=(D(xf))

            # min predf
            lossf=(predf.mean())

            # gradient penalty
            gp = gradient_penalty(D, xr, xf)

            loss_D = lossr + lossf + gp
            optim_D.zero_grad()
            loss_D.backward()
            # for p in D.parameters():
            #     print(p.grad.norm())
            optim_D.step()

            # 2. train Generator
            z = torch.randn(batch_size, 2).cuda()
            xf = G(z)
            predf = (D(xf))
            # max predf
            loss_G = - (predf.mean())
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()

            if epoch % 100 == 0:
                viz.line([[loss_D.item(), loss_G.item()]], [epoch], win='loss', update='append')

                generate_image(D, G, xr, epoch)

                print(loss_D.item(), loss_G.item())

if __name__ == '__main__':
    main()