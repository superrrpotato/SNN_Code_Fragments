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
n_bins = 20
u_min = -2
u_max = 1
bin_length = (u_max-u_min)/n_bins
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
    scale = 0.2
    w_e = scale * (np.random.rand(N)-0.5)
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
    
plt.figure(figsize=(20,15))
x_record=[]
y_record=[]
for i in range(n_bins):
    plt.subplot(4,5,i+1)
    x = np.array(correlation_dict[i])[:,0]
    y = np.array(correlation_dict[i])[:,1]
#     my_pwlf = pwlf.PiecewiseLinFit(x, y)
#     breaks = my_pwlf.fit(2)
    
    poly_num = 1
    coef = np.polyfit(x, y, poly_num)
    interval = 10
    x_fit = np.arange(min(x),max(x),(max(x)-min(x))/interval)
    y_fit = np.polyval(coef,x_fit)
    print(coef[0])
#     y_fit = my_pwlf.predict(x_fit)
    x_record += [x_fit]
    y_record += [y_fit]
    plt.scatter(x,y, c='black',alpha=.1,marker='+')
    plt.plot(x_fit,y_fit,c='r')
    plt.title('membrane potential:'+str(round(i*bin_length+u_min, 2)))
plt.figure()
plt.scatter(i_e*(i>0.1), e*(i>0.1), c='black',alpha=.001,marker='+')
for i in range(n_bins):
    plt.plot(x_record[i],y_record[i])
