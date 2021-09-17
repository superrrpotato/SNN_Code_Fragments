import numpy as np
import matplotlib.pyplot as plt
# import pwlf
%matplotlib inline
# Total simulation time /ms
T=500
# Membrane potential time constant
tau_u = 50
# Postsynaptic current time constant
tau_s = 20

u = np.zeros(T)
s = np.zeros(T)
a = np.zeros(T)

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

# random input spikes
# num of inputs:
N = 50
# spike probability
p = 0.05
# generate

rounds = 1000
fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
n_bins = 50
u_min = -2
u_max = 1
bin_length = (u_max-u_min)/n_bins
repeat_num = 20
k_record = {}
for e_s in [1, 0.5, 0.1, 0.05, 0.01]:
    print(e_s)
    k = []
    for j in range(n_bins+1):
        k += [[]]
    for k_stat in range(repeat_num):
        
        correlation_dict = []
        for j in range(n_bins+1):
            correlation_dict += [[]]
        for q in range(rounds):
        #     print(p)
            s_i = np.random.rand(N,T)<p
            a_i = psc(s_i,tau_s)


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

            u_e = np.zeros(T)
            s_e = np.zeros(T)
            a_e = np.zeros(T)

            s_ei = np.random.rand(N,T)<p
            a_ei = psc(s_ei,tau_s)
            e_scale = e_s
            w_e = e_scale * (np.random.rand(N)-0.5)
            i_e = w_e.dot(a_ei)
            # Forward
            u_e_temp = 0
            for t in range(T):
                u_e[t] = u_e_temp * (1-1/tau_u) + i[t] + i_e[t]
                s_e[t] = (u_e[t]>1).astype('float')
                u_e_temp = u_e[t] * (1-s_e[t])
            a_e = psc(s_e, tau_s)

            e = a_e-a
            for j_index, j in enumerate(np.clip(((u - u_min)/bin_length).astype(int), 0, n_bins).tolist()):
                correlation_dict[j] += [[i_e[j_index], e[j_index]]]

        for i in range(n_bins+1):
            x = np.array(correlation_dict[i])[:,0]
            y = np.array(correlation_dict[i])[:,1]
            poly_num = 1
            coef = np.polyfit(x, y, poly_num)
            k[i] += [coef[0]]
    #         print(coef[0])
    k_record[e_s] = k
    k = np.array(k).T/np.max(k)
    for k_stat in range(repeat_num):
        plt.scatter(u_x, k[k_stat], marker='+', color='black', alpha=.05)
    meam = np.mean(k,axis=0)
    std = np.std(k,axis=0)
    plt.plot(u_x, meam, label='scale='+str(e_scale))
    plt.fill_between(u_x, meam-std, meam+std,alpha=.4)
