import numpy as np
from wave_equation import wave_equation


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

derivatives = wave_equation(state, sigma, h)

print("dudt center:", derivatives[0, 50])
print("dvdt center:", derivatives[1, 50])

print("x =", x[49])
print("u =", state[0, 49])
print("dudt =", derivatives[0, 49])
print("dvdt =", derivatives[1, 49])