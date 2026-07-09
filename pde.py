import numpy as np

def equation(u, sigma, dx):
    dudt = np.zeros_like(u) #initialize the array for the derivatives

    # boundary conditions
    dudt[0] = -sigma * (u[0] - u[-1]) / dx

    for i in range(1, len(u)-1): #compute each point's derivative using the second spatial derivative
        dudt[i] = -sigma * (
            u[i] - u[i-1]
        ) / dx

    return dudt