import numpy as np

def wave_equation(state, sigma, h): #state must be a 2d array with u and v stacked on top of each other

    u = state[0]
    v = state[1]

    dudt = np.zeros_like(u)
    dvdt = np.zeros_like(v)

    for i in range(1, len(u)-1):
        dudt[i] = v[i] #du/dt=v

        dvdt[i] = sigma**2 * (  #du^2/dx^2
            u[i+1] - 2*u[i] + u[i-1]
        ) / h**2

    return np.array([dudt, dvdt])