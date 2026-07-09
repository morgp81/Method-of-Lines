import numpy as np
import matplotlib.pyplot as plt
from heat_equation import heat_equation
from rk_methods import euler, explicit_rk2, implicit_rk2

# Parameters
N = 100  # number of spatial points
delta_x = 1/N  # spatial step size
sigma = 0.0001  # thermal diffusivity

steps = 5000  # number of time steps
dt = 0.025  # time step size

x = np.linspace(0, 1, N+1) # initial condition: sine wave
u = np.sin(2*np.pi*x) 

values = [u.copy()] # store the values at each time step for plotting

for t in range(steps): #time loop
    u=implicit_rk2(u, heat_equation, dt, sigma, delta_x)
    values.append(u.copy())

values = np.array(values) 
print(values)

plt.imshow(values, aspect='auto', extent=[0, 1, 0, steps*dt], origin='lower')
plt.colorbar(label='Temperature')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('Heat Equation Evolution')
plt.savefig("heat_equation2d.png")
plt.show()


