import numpy as np

def bracket_minimum(f, x=0, s=1e-2, k=2.0):
    a, ya = x, f(x)
    b, yb = a + s, f(a + s)
    if yb > ya:
        a, b = b, a
        ya, yb = yb, ya
        s = -s
    while 1:
        c, yc = b + s, f(b + s)
        if yc > yb:
            return (a, c) if a < c else (c, a)
        a, ya, b, yb = b, yb, c, yc
        s *= k

def line_search(f, x, d):
    def objective(z):
        return f(x + z*d)
    a, b = bracket_minimum(objective)
    z = golden_section_search_minimize(objective, a, b, .002)
    return x + z*d

def golden_section_search_minimize(f, a, b, min, n = 10000):
    ϕ = (1 + 5 ** 0.5) / 2
    ρ = ϕ-1
    d = ρ * b + (1 - ρ)*a
    yd = f(d)
    count = 0
    while abs(a - b) > min and count < n:
        c = ρ*a + (1 - ρ)*b
        yc = f(c)
        if yc < yd:
            b, d, yd = d, c, yc
        else:
            a, b = b, c
        count = count + 1
    return (a+b)/2

class BFGS:
    def __init__(self, f, del_f, x):
        m = len(x)
        self.Q = np.identity(m)
        self.f = f
        self.del_f = del_f
        return

    def step(self, x):
        Q, g = self.Q, self.del_f(x)
        x_prime = line_search(self.f, x, np.dot(-Q,g))
        g_prime = self.del_f(x_prime)
        δ = x_prime - x
        γ = g_prime - g

        self.Q = Q - (δ*γ.T*Q + Q*γ*δ.T)/(δ.T*γ) + (1 + (γ.T*Q*γ)/(δ.T*γ))[1]*(δ*δ.T)/(δ.T*γ)
        return x_prime


### test functions
def quad(x, vals = [.1,.2,.3,.4,.5,.6,.232,.9,.98,.0012]):
    sum = 0
    for i in range(len(vals)):

        sum += (x[i]-vals[i])**2
    return sum

def del_quad(x, vals = [.1,.2,.3,.4,.5,.6,.232,.9,.98,.0012]):
    del_q = []
    for i in range(len(vals)):
        del_q.append(2*(x[i] - vals[i]))
    return np.asarray(del_q)
