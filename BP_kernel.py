window_length = 600
# tau_p = 50
tau_m = 50
tau_s = 50
plt.figure(figsize=(16,10))
for tau_s in [20,60,100]:
    window_t = np.arange(-window_length,window_length+1,1)
    mem_k = 1/tau_m * np.exp(-window_t/(tau_m))*(window_t>=0) #+ -0.4* np.exp(window_t/(tau_m)) * (window_t<0)
    mem_k[window_length] *= 0.5
    psc_k = 1/tau_s * np.exp(-window_t/(tau_s))*(window_t>=0)
    psc_k[window_length] *= 0.5
    BP_k = np.convolve(mem_k, psc_k)[window_length:-window_length]
    BP_kk = np.convolve(psc_k, BP_k)[window_length:-window_length]
    
    # plt.plot(window_t, BP_k,label='BP kernel', lw=10)
    # plt.plot(window_t, BP_kk,label='BP s kernel', lw=10)
    calculated1 = (np.exp(-window_t/(tau_m))-np.exp(-window_t/(tau_s)))*(window_t>=0)/(tau_m-tau_s)
    calculated2 = tau_m/((tau_m-tau_s)**2) *(np.exp(-window_t/(tau_m))*(window_t>=0)-np.exp(-window_t/(tau_s))*(window_t>=0))-window_t/(tau_s*(tau_m-tau_s)) * np.exp(-window_t/(tau_s))*(window_t>=0)

    # plt.plot(window_t, calculated1,label='BP s kernel', lw=4)
    plt.plot(window_t[window_length-50:], calculated2[window_length-50:],label='tau_m=50, tau_s='+str(tau_s), lw=10)
tau_m = 50
tau_s = 50
for tau_m in [20,60,100]:
    window_t = np.arange(-window_length,window_length+1,1)
    mem_k = 1/tau_m * np.exp(-window_t/(tau_m))*(window_t>=0) #+ -0.4* np.exp(window_t/(tau_m)) * (window_t<0)
    mem_k[window_length] *= 0.5
    psc_k = 1/tau_s * np.exp(-window_t/(tau_s))*(window_t>=0)
    psc_k[window_length] *= 0.5
    BP_k = np.convolve(mem_k, psc_k)[window_length:-window_length]
    BP_kk = np.convolve(psc_k, BP_k)[window_length:-window_length]
    
    # plt.plot(window_t, BP_k,label='BP kernel', lw=10)
    # plt.plot(window_t, BP_kk,label='BP s kernel', lw=10)
    calculated1 = (np.exp(-window_t/(tau_m))-np.exp(-window_t/(tau_s)))*(window_t>=0)/(tau_m-tau_s)
    calculated2 = tau_m/((tau_m-tau_s)**2) *(np.exp(-window_t/(tau_m))*(window_t>=0)-np.exp(-window_t/(tau_s))*(window_t>=0))-window_t/(tau_s*(tau_m-tau_s)) * np.exp(-window_t/(tau_s))*(window_t>=0)

    # plt.plot(window_t, calculated1,label='BP s kernel', lw=4)
    plt.plot(window_t[window_length-50:], calculated2[window_length-50:],':',label='tau_s=50, tau_m='+str(tau_m), lw=10)
plt.legend()
    
