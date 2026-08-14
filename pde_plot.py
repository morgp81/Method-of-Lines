import numpy as np
import matplotlib.pyplot as plt
from pde import equation
from rk_methods import euler, explicit_rk2, implicit_rk2

# Parameters
N = 400  # number of spatial points
delta_x = 1/N  # spatial step size
sigma = 1.0

steps = 200  # number of time steps
dt = delta_x/2*sigma  # time step size

x = np.linspace(0, 1, N, endpoint=False) # initial condition: sine wave
u = np.sin(2*np.pi*x) 

values = [u.copy()] # store the values at each time step for plotting

for t in range(steps): #time loop
    u=implicit_rk2(u, equation, dt, sigma, delta_x)
    values.append(u.copy())

finalTime=dt*steps


values = np.array(values) 
print(values[len(values)-1]) #print the final values
print(len(values))
for i in range(len(values)):
    print(i,len(values[i]))

numberOfPoints=len(values[0])
dataPoints=np.zeros(numberOfPoints)
for i in range(numberOfPoints):
    xi=delta_x*i
    dataPoints[i] = np.sin(2*np.pi*(xi-sigma*finalTime))


plt.imshow(values, aspect='auto', extent=[0, 1, 0, steps*dt], origin='lower')
plt.colorbar(label='U value')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('PDE Evolution')
plt.show()

# plt.plot(x, values[len(values)-1], label='Final State')
# plt.plot(x, dataPoints, label='Analytical Solution', linestyle='dashed')
# plt.xlabel('Position')
# plt.ylabel('U value')
# plt.title('Final State of the PDE')
# plt.legend()
# plt.show()




