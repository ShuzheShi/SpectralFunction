import time
import math
import os
import numpy as np
from pickle import LIST
import matplotlib.pyplot as plt

import torch
# from torch._C import ThroughputBenchmark
import torch.optim as optim
import torch.nn as nn

# initialization
import ini
device = ini.initial("NN")
from paras import args

# setting default rho and D
omegal = 500
omega_up = 20
domegai = omega_up/omegal
omegai = torch.linspace(domegai,omega_up,steps=omegal).to(device) #,requires_grad=True

taul = 25
tau_up = 100
dtaui = tau_up/taul
taui = torch.linspace(0.001 , tau_up - dtaui - 0.001,steps=taul).to(device)


input = torch.ones(1).to(device) # ,requires_grad=True

def chi2(pre,obs): 
    out = (obs - pre)**2
    out = out.sum()
    return out

def dl(omegai,rhoi,obs,pre):    
    omegai = omegai.reshape(1,-1)
    rhoi = rhoi.reshape(1,omegal)
    tauii = taui.reshape(len(taui),1).to(device)
    matri = torch.ones(len(taui),omegal).to(device)
    tauii = tauii* matri

    fi = torch.log(torch.exp(rhoi) - 1)

    dsigma = 1/(1+torch.exp(-fi))
    lhs = args.l2 * fi/dsigma /((fi**2).sum())**((args.depth )/(args.depth + 1))

    deltai = ((obs - pre)).reshape(len(taui),1) * matri
    rhs = deltai * (omegai / (tauii**2 + omegai**2)/math.pi)

    out = (lhs - rhs.sum(dim = 0))**2
    # out = (lhs / rhs.sum(dim = 0) - 1)**2

    out = torch.sum(out,dim=1)
    return out, lhs, rhs


def D(taui,omegai,rhoi):
    # taui = taui #.reshape(-1)
    omegai = omegai.reshape(1,-1)
    rhoi = rhoi.reshape(1,omegal)

    tauii = taui.reshape(len(taui),1).to(device)
    matri = torch.ones(len(taui),omegal).to(device)
    tauii = tauii* matri

    out = (omegai / (tauii**2 + omegai**2)/math.pi)*rhoi*domegai 
    # #  out = torch.exp(-taui*omegai)*rhoi*domegai
    out = torch.sum(out,dim=1)
    return out

def Dp(taui,omegai,rhoi): # partial D(tau)
    taui = taui.reshape(-1)
    omegai = omegai.reshape(1,-1)
    rhoi = rhoi.reshape(1,omegal)

    tauii = taui.reshape(len(taui),1).to(device)
    tauii = tauii* torch.ones(len(taui),omegal).to(device)

    out = - (2* omegai * tauii / (tauii**2 + omegai**2) **2 /math.pi)*rhoi*domegai 
    # #  out = torch.exp(-taui*omegai)*rhoi*domegai
    out = torch.sum(out,dim=1)
    return out

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.001)

class Net(nn.Module):
    def __init__(self,):
        super(Net, self).__init__()

        self.input = torch.nn.Sequential(nn.Linear(1,args.width,bias=False) )
        latents = []
        for i in range(args.depth - 1):
            latents.append(torch.nn.Sequential(nn.Linear(args.width,args.width,bias=False))) # ,nn.ReLU()
        self.latent =  torch.nn.Sequential(*latents)
        self.output = torch.nn.Sequential(nn.Linear(args.width,omegal,bias=False),nn.Softplus())
        # self.output = torch.nn.Sequential(nn.Linear(args.width,omegal,bias=False))


    def forward(self, x):
        z = self.input(x)
        y = self.latent(z)
        output = self.output(y)
        return output

class NNList(nn.Module):
    def __init__(self, ):
        super(NNList, self).__init__()

        self.layers = torch.nn.Sequential(
            nn.Linear(1,omegal,bias=False),nn.Softplus()
            )

    def forward(self, x):
        output = self.layers(x)
        return output

if args.depth == 0:
    nnrho = NNList()
else:
    nnrho = Net()

nnrho.apply(init_weights)
nnrho = nnrho.to(device)

print(nnrho)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
print('parameters_count:',count_parameters(nnrho))

#############################################################################################
# import  mock  data #
######################
directory='./data/fig3'
data = np.loadtxt('{}/True_D.txt'.format(directory), delimiter='\t', unpack=True)

taui =  torch.from_numpy(data[0]).float().to(device)
target =  torch.from_numpy(data[1]).float().to(device)

taul = len(taui)
tau_up = max(taui)
dtaui = tau_up/taul

taui_p = taui[:-1] + dtaui/2
target_p = (target[1:] - target[:-1])/dtaui

out = (nnrho(input))
out_old = out

t = time.time()

#############################################################################################
# warm up #
######################
if os.path.exists('nnRho{}_{}_{}_{}_d{}'.format(args.Index,args.noise,args.epochs,"linear",args.depth)):
    nnrho = torch.load('nnRho{}_{}_{}_{}_d{}'.format(args.Index,args.noise,args.epochs,"linear",args.depth))
else:
    optimizer = optim.Adam(nnrho.parameters(), lr=args.lr)#, weight_decay = 0) # 1e-3
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.1)
    l2_lambda = 0.01
    slambda = args.slambda
    nnrho.zero_grad()
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i  in range(25):
            # get the inputs; data is a list of [inputs, labels]
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            rhoi = (nnrho(input))
            outputs = D(taui,omegai,rhoi)
            # outputs_p = Dp(taui_p,omegai,rhoi)
            loss = chi2(outputs, target) #+  chi2(outputs_p, target_p)
            chis = loss.item()

            l2 = l2_lambda * sum([(p**2).sum() for p in nnrho.parameters()])
            loss+= l2

            loss+= ((slambda*((rhoi[1:]-rhoi[:-1])**2).sum()))

            running_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

            if (i % 25 == 24)&(epoch % 50 == 49) :    # print every 50 epoches
                print('[%d, %5d] chi2: %.8e lr: %.5f loss: %8f' % 
                    (epoch + 1, i + 1, chis, optimizer.param_groups[0]['lr'],loss.item()))

        running_loss = 0.0

        if (epoch % 200 == 199) : 
            slambda *= 0.1
            l2_lambda = args.l2*domegai

        if (epoch % 400 == 399) : 
            slambda = args.slambda
            l2_lambda = 0.01
    
    torch.save(nnrho, 'nnRho{}_{}_{}_{}_d{}'.format(args.Index,args.noise,args.epochs,"linear",args.depth))

#############################################################################################
# training #
###########
l2_lambda = args.l2*domegai
optimizer = optim.Adam(nnrho.parameters(), lr=args.lr*0.1, weight_decay = 0) # 1e-3
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5)
nnrho.zero_grad()
rec_loss,steps, dlv = 0, 0, 0
for epoch in range(args.epochs*40):  # loop over the dataset multiple times
    running_loss = 0.0
    for i  in range(25):
        # get the inputs; data is a list of [inputs, labels]
        # zero the parameter gradients
        optimizer.zero_grad()
        
        # forward + backward + optimize
        rhoi = (nnrho(input))
        outputs = D(taui,omegai,rhoi)
        # outputs_p = Dp(taui_p,omegai,rhoi)
        loss = chi2(outputs, target) #+  chi2(outputs_p, target_p)
        chis = loss.item()

        l2 = l2_lambda * sum([(p**2).sum() for p in nnrho.parameters()])
        loss+= l2

        dl_v, _, _ = dl(omegai,rhoi,target,outputs)
        if (i + epoch*25)%1 ==0 and (i + epoch*25)<20:
            rec_loss = np.append(rec_loss,loss.item())
            steps  = np.append(steps, i + epoch*25)
            dlv  = np.append(dlv, dl_v.item())
        if (i + epoch*25)%10 ==0 and np.log10(i + epoch*25)<2:
            rec_loss = np.append(rec_loss,loss.item())
            steps  = np.append(steps, i + epoch*25)        
            dlv  = np.append(dlv, dl_v.item())
        if (i + epoch*25)%100 ==0 and np.log10(i + epoch*25)<3:
            rec_loss = np.append(rec_loss,loss.item())
            steps  = np.append(steps, i + epoch*25)      
            dlv  = np.append(dlv, dl_v.item())
        if (i + epoch*25)%1000 ==0:
            rec_loss = np.append(rec_loss,loss.item())
            steps  = np.append(steps, i + epoch*25)      
            dlv  = np.append(dlv, dl_v.item())
        # if (i + epoch*25)%10000 ==0 :
        #     rec_loss = np.append(rec_loss,loss.item())
        #     steps  = np.append(steps, i + epoch*25)     
        #     dlv  = np.append(dlv, dl_v.item())

        running_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i % 25 == 24)&(epoch % 50 == 49) :    # print every 20 epoches
            print('[%d, %5d] chi2: %.8e lr: %.10f loss: %.8e' % 
                (epoch + 1 + args.epochs, i + 1, chis, optimizer.param_groups[0]['lr'],loss.item()))
    
    # if loss< 10**(-args.noise * 2 - 1):
    #     print('Early stopping!' )
    #     break

    # if epoch > args.maxiter and (running_loss/25 - loss.item())>= 0:
    #     print('Early stopping!' )
    #     break

    running_loss = 0.0

dl_v, lhs , rhs = dl(omegai,rhoi,target,outputs)

print('Finished Training')
elapsed = time.time() - t
print('Cost Time = ',elapsed)

#############################################################################################
# plot figures and export #
############

out = (nnrho(input))
np.savetxt('{}/Rec_rho_NN_d{}_1e-06.txt'.format(directory, args.depth), \
         np.column_stack((omegai.cpu().detach().numpy(),out.cpu().detach().numpy())), fmt='%8f\t%.10f') 
np.savetxt('{}/test_{}_d{}.txt'.format(directory,"linear",args.depth), \
    np.column_stack((lhs.reshape(-1).detach().numpy(),rhs.sum(dim = 0).reshape(-1).detach().numpy())),fmt='%10e %10e') 
