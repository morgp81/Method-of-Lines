import numpy as np
import matplotlib.pyplot as plt
from pde import equation
from rk_methods import euler, explicit_rk2, implicit_rk2

N = 4
dx = 1/N
x = np.linspace(0, 1, N, endpoint=False)
u = np.sin(2*np.pi*x)

print(u)
print(equation(u, 1, dx))