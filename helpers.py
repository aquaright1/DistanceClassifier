### Helper functions ###
import scipy as sp

def gamma_mle(data, iterations = 4):
    #using Gamma(shape,scale) not Gamma(shape, rate)
    alpha = 0 # 0 is k, 1 is theta
    x = np.asarray([0,0]) #0 is np.log(np.mean(x)) 1 is np.mean(np.log(x))

    x[0] = np.log(np.mean(data))
    x[1] = np.mean(np.log(data))

    alpha= .5/(x[0] - x[1])

    k = alpha
    for i in range(iterations):
        digamma = sp.special.digamma(k)
        digamma_prime = sp.special.polygamma(1, k)
        k = 1/ (1/k + (x[1] - x[0] + np.log(k) - digamma)/(k**2*(1/k - digamma_prime)))
        ##print("itermidiary step:", k)

    alpha[0] = k
    alpha[1] = np.exp(x[0])/alphas[0]
    return alpha
