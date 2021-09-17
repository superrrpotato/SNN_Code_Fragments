import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# Total simulation time /ms
T=1000
# Membrane potential time constant
tau_u = 50
# Postsynaptic current time constant
tau_s = 20

u = np.zeros(T)
s = np.zeros(T)
a = np.zeros(T)



# random input spikes
# num of inputs:
N = 50
# spike probability
p = 0.05
# generate
s_i = np.random.rand(N,T)<p

def psc(s, tau):
    if len(s.shape)==2:
        a = np.zeros(shape=(N,T))
        a_temp = np.zeros(N)
    else:
        a = np.zeros(T)
        a_temp = 0.
    for t in range(T):
        if len(s.shape)==2:
            a[:,t] = a_temp * (1-1/tau) + (1/tau) * s[:,t].astype("float")
            a_temp = a[:,t]
        else:
            a[t] = a_temp * (1-1/tau) + (1/tau) * s[t].astype("float")
            a_temp = a[t]
    return a
a_i = psc(s_i,tau_s)

while (np.sum(s)<10) or (np.sum(s)>30):
    # weight init
    scale = 1
    bias = 0.00
    w = scale * (np.random.rand(N)-0.5) + bias

    # input
    i = w.dot(a_i)

    # Forward
    u_temp = 0
    for t in range(T):
        u[t] = u_temp * (1-1/tau_u) + i[t]
        s[t] = (u[t]>1).astype('float')
        u_temp = u[t] * (1-s[t])
    a = psc(s, tau_s)

plt.figure(figsize=(15,5))
plt.imshow(s_i,cmap='binary')
plt.title('input spike trains')
plt.figure(figsize=(15,5))
plt.imshow(a_i,cmap='binary')
plt.title('input postsynaptic currents (PSCs)')
plt.figure(figsize=(14.8,2))
plt.plot(i, c='black')
plt.title('Total input current = weighted sum of PSCs')
plt.figure(figsize=(14.8,2))
plt.plot(s, c='black',label='output spike train')
plt.plot(u, label='membrane potential')
plt.plot(a*5, c='red', label='output postsynaptic current')
plt.legend()
plt.title('Somatic waveforms')


u_e = np.zeros(T)
s_e = np.zeros(T)
a_e = np.zeros(T)

s_ei = np.random.rand(N,T)<p
a_ei = psc(s_ei,tau_s)
scale = 0.6
w_e = scale * (np.random.rand(N)-0.5)
i_e = w_e.dot(a_ei)
# Forward
u_e_temp = 0
for t in range(T):
    u_e[t] = u_e_temp * (1-1/tau_u) + i[t] + i_e[t]
    s_e[t] = (u_e[t]>1).astype('float')
    u_e_temp = u_e[t] * (1-s_e[t])
a_e = psc(s_e, tau_s)

plt.figure(figsize=(15,5))
plt.imshow(s_i,cmap='binary')
plt.title('input spike trains')
plt.figure(figsize=(15,5))
plt.imshow(a_i,cmap='binary')
plt.title('input postsynaptic currents (PSCs)')

plt.figure(figsize=(15,5))
plt.imshow(s_ei,cmap='binary')
plt.title('input error spike trains')
plt.figure(figsize=(15,5))
plt.imshow(a_ei,cmap='binary')
plt.title('input error postsynaptic currents (PSCs)')

plt.figure(figsize=(14.8,2))
plt.plot(i, c='black',label='input PSCs')
plt.plot(i_e, c='green',label='input error PSCs')
plt.plot(i+i_e, c='red',label='sum of two')
plt.legend()
plt.title('Total input current (= weighted sum of PSCs)')

plt.figure(figsize=(14.8,2))
plt.plot(s, c='black',label='output spike train')
plt.plot(u, label='membrane potential')
plt.plot(a*5, c='red', label='output postsynaptic current')
plt.legend()
plt.title('Somatic waveforms')

plt.figure(figsize=(14.8,2))
plt.plot(s_e, c='black',label='output spike train')
plt.plot(u_e, label='membrane potential')
plt.plot(a_e*5, c='red', label='output postsynaptic current')
plt.legend()
plt.title('Somatic waveforms with error added')

plt.figure(figsize=(14.8,2))
plt.plot(i_e, c='green',label='input error PSCs')
plt.plot((a_e-a), c='red', label='output postsynaptic current difference')
plt.legend()
plt.title('error - PSCs difference alignment')
