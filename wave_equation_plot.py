import numpy as np
import matplotlib.pyplot as plt
from wave_equation import wave_equation
from rk_methods import euler, explicit_rk2, implicit_rk2


# spatial grid
N = 101
x = np.linspace(0, 1, N)
h = 1 / (N - 1)

# wave speed
sigma = 1

# initial pulse parameters
x0 = 0.5
pulse_width = 0.1

# initial conditions
u = np.exp(-((x - x0) / pulse_width)**2)
v = np.zeros_like(x)

# Dirichlet boundary condition
u[0] = 0
u[-1] = 0
v[0] = 0
v[-1] = 0

state = np.array([u, v])

values = [u.copy()]
dt = 0.001
steps = 1000

for step in range(steps):
    state = implicit_rk2(state, wave_equation, dt, sigma, h)

    values.append(state[0].copy())

plt.imshow(values, aspect='auto', extent=[0, 1, 0, steps*dt], origin='lower')
plt.xlabel("Position")
plt.ylabel("Time")
plt.title("Wave Equation")
plt.colorbar(label="u(x,t)")
plt.show()

