import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from IPython.display import clear_output
%matplotlib qt5

total_time = 1000
N = 1000
tau_s = 50
# a = np.zeros(N)
t = np.arange(N)
a = 1/tau_s*np.exp(-(t-N/2)/tau_s) *(t>=N/2)
b = 1/tau_s*np.exp((t-N/2)/tau_s) *(t<=N/2)
c = np.convolve(a,b)[int(N/2): -int(N/2)+1]


kernel = torch.tensor(c).view(1,1,1000).float()
new_stdp = nn.Conv1d(1, 1, 1000, padding=1000)
torch.nn.init.constant_(new_stdp.weight.data, 0.)
torch.nn.init.constant_(new_stdp.bias.data, 0.)
optimizer = torch.optim. AdamW(new_stdp.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1)
t = torch.arange(2001)
N=2000
tau_m = 50
tau_s = 20
b = 1/tau_s* (torch.exp(-(t-N/2)/tau_m) *(t>=1000) + -0.0* torch.exp((t-N/2)/(tau_m)) * (t<1000))
plt.plot(b)
# b = torch.tensor(b).view(1,1000,1)
loss_record=[]

plt.figure(figsize=(10,3))
for i in range(1000):
#     print(new_stdp(kernel).view(2001)[501:-500].shape)
    output = new_stdp(kernel).view(2001)
    loss = torch.sum((output - b)**2)
    loss_record += [loss.detach().numpy().tolist()]
    optimizer.zero_grad()
    loss.backward()
    if i % 10 == 0:
        clear_output(wait=True)
#         new_stdp.weight.value[:400]=0.
#         new_stdp.weight.value[600:]=0.
        plt.clf()
        
        plt.subplot(1,2,1)
        plt.title('output fitting curve')
        plt.plot(b)
        plt.plot(output.detach().numpy(),c='r')
        plt.subplot(1,2,2)
        plt.title('loss='+str(loss.detach().numpy().tolist()))
        plt.plot(new_stdp.weight.view(1000).detach().numpy())
        plt.pause(0.05)
#         plt.plot(new_stdp.weight.view(1000).detach().numpy(),alpha=.1)
    optimizer.step()
plt.figure()
plt.plot(loss_record)
plt.yscale('log')
