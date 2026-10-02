import numpy as np
from rk_methods import euler, explicit_rk2, implicit_rk2
import matplotlib.pyplot as plt

def burgers_equation(u, dx, points):

    dudt = np.zeros_like(u)

    flux = u**2 * 0.5

    dudt[0] = -(flux[1] - flux[-1]) / (2*dx)

    for i in range(1, points-1):
        dudt[i] = -(flux[i+1] - flux[i-1]) / (2*dx)

    dudt[-1] = -(flux[0] - flux[-2]) / (2*dx)

    return dudt

N = 100
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]

T = 0.2
dt = 0.001
steps = int(T/dt)

initial_values = np.sin(2 * np.pi * x)

values_mol = initial_values.copy()
solution_mol = np.zeros((steps + 1, N))
solution_mol[0] = values_mol

# MOL loop

for step in range(steps):

    values_mol = implicit_rk2(
        values_mol,
        burgers_equation,
        dt,
        dx,
        N
    )

    solution_mol[step + 1] = values_mol

def lax_friedrichs(u, dx, dt, points):
    new_u = np.zeros_like(u)

    flux = u**2 * 0.5

    new_u[0] = 0.5 * (u[1] + u[-1]) - dt / (2 * dx) * (flux[1] - flux[-1])

    for i in range(1, points - 1):
        new_u[i] = 0.5 * (u[i+1] + u[i-1]) - dt / (2*dx) * (flux[i+1] - flux[i-1])
    new_u[-1] = 0.5 * (u[0] + u[-2]) - dt / (2 * dx) * (flux[0] - flux[-2])

    return new_u

values_lax = np.sin(2 * np.pi * x)

solution_lax = np.zeros((steps + 1, N))
solution_lax[0] = values_lax

for step in range(steps):
    values_lax = lax_friedrichs(values_lax, dx, dt, N)
    solution_lax[step + 1] = values_lax

plt.figure(figsize=(10, 6))
plt.imshow(solution_mol, aspect='auto', extent=[0, 1, 0, T], origin='lower')
plt.colorbar(label='u(x,t)')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('Burgers Equation Evolution MOL')
plt.savefig("burgers_equation.png")
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(solution_lax, aspect='auto', extent=[0, 1, 0, T], origin='lower')
plt.colorbar(label='u(x,t)')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('Burgers Equation Evolution Lax-Friedrichs')
plt.savefig("burgers_equation_lax.png")
plt.show()

plt.plot(x, initial_values, label="Initial")
plt.plot(x, solution_mol[-1], label="MOL")
plt.plot(x, solution_lax[-1], label="Lax-Friedrichs")

plt.xlabel("Position")
plt.ylabel("u(x,t)")
plt.title("Burgers Equation at t = 0.2")
plt.legend()
plt.savefig("burgers_equation_final.png")
plt.show()