import numpy as np
from rk_methods import euler, explicit_rk2, implicit_rk2
import matplotlib.pyplot as plt
from godunov import godunov

def burgers_equation(u, dx, points):

    dudt = np.zeros_like(u)

    flux = u**2 * 0.5

    dudt[0] = -(flux[1] - flux[-1]) / (2*dx)

    for i in range(1, points-1):
        dudt[i] = -(flux[i+1] - flux[i-1]) / (2*dx)

    dudt[-1] = -(flux[0] - flux[-2]) / (2*dx)

    return dudt

N = 1000
x = np.linspace(0, 1, N, endpoint=False)
dx = x[1] - x[0]

T = 0.2
dt = 0.0005
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

values_lax = initial_values.copy()

solution_lax = np.zeros((steps + 1, N))
solution_lax[0] = values_lax

for step in range(steps):
    values_lax = lax_friedrichs(values_lax, dx, dt, N)
    solution_lax[step + 1] = values_lax

values_godunov = initial_values.copy()

solution_godunov = np.zeros((steps + 1, N))
solution_godunov[0] = values_godunov

for step in range(steps):

    flux_godunov = np.zeros(N)

    for i in range(N):
        uL = values_godunov[i]
        uR = values_godunov[(i + 1) % N]

        flux_godunov[i] = godunov(uL, uR)

    new_values_godunov = np.zeros_like(values_godunov)

    for i in range(N):

        dudt_i = -(flux_godunov[i] - flux_godunov[i - 1]) / dx

        new_values_godunov[i] = values_godunov[i] + dt * dudt_i

    values_godunov = new_values_godunov

    solution_godunov[step + 1] = values_godunov

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

plt.figure(figsize=(10, 6))
plt.imshow(solution_godunov, aspect='auto', extent=[0, 1, 0, T], origin='lower')
plt.colorbar(label='u(x,t)')
plt.xlabel('Position')
plt.ylabel('Time')
plt.title('Burgers Equation Evolution Godunov')
plt.savefig("burgers_equation_godunov.png")
plt.show()

plt.plot(x, initial_values, label="Initial")
plt.plot(x, solution_mol[0], label="MOL 0.0")
plt.plot(x, solution_lax[0], label="Lax-Friedrichs 0.0")
plt.plot(x, solution_godunov[0], label="Godunov 0.0")
plt.plot(x, solution_mol[-1], label="MOL 0.2")
plt.plot(x, solution_lax[-1], label="Lax-Friedrichs 0.2")
plt.plot(x, solution_godunov[-1], label="Godunov 0.2")

plt.xlabel("Position")
plt.ylabel("u(x,t)")
plt.title("Burgers Equation at t = 0 and 0.2")
plt.legend()
plt.savefig("burgers_equation_final.png")
plt.show()