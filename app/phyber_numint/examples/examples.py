from __future__ import annotations
import numpy as np
from typing import Tuple, Union

def lorenz_system(t: float, X: Union[Tuple[float,...], np.ndarray], beta: float, sigma: float, rho: float) -> Tuple[float,float,float]:
    '''
        x' = sigma(y - x)
        y' = x(rho - z) - y
        z' = x*y - beta*z
    '''
    x, y, z = X
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return (dxdt, dydt, dzdt)

def damped_oscillator(t: float, X: Union[Tuple[float,...], np.ndarray], k: float, nu: float, m: float=1) -> Tuple[float,float]:
    '''
        mx'' + nu x' + k x = 0

        u' = v
        v' = -(nu/m) v - (k / m) u
    '''
    u, v = X
    u_p = v
    v_p = -(nu / m) * v - (k / m) * u
    return (u_p, v_p)

def damped_oscillator_analytical(t: Union[float, np.ndarray], X0: Tuple[float, float], k: float, nu: float, m: float=1) -> np.ndarray:
    '''
        Analytical solution to the damped harmonic oscillator.

        Parameters:
        - t: Time values where the solution is evaluated.
        - m: Mass of the oscillator.
        - b: Damping coefficient.
        - k: Spring constant.
        - X0 = (A, B)
            - A: Constant determined by initial position.
            - B: Constant determined by initial velocity.

        Returns:
        - x: Position as a function of time.
    '''
    A, B = X0
    omega_d = np.sqrt(k/m - (nu/(2*m))**2)
    x = np.exp(-nu/(2*m) * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    return x

def damped_oscillator_2d_analytical(t: Union[float, np.ndarray], X0: Tuple[float, float, float, float], k: float, nu: float, m: float=1) -> np.ndarray:
    '''
        Analytical solution to the damped harmonic oscillator.

        Parameters:
        - t: Time values where the solution is evaluated.
        - m: Mass of the oscillator.
        - b: Damping coefficient.
        - k: Spring constant.
        - X0 = (A1, A2, B1, B2)
            - A1: Constant determined by initial x position.
            - A2: Constant determined by initial y position.
            - B1: Constant determined by initial x velocity.
            - B2: Constant determined by initial y velocity.

        Returns:
        - x: Position as a function of time.
    '''
    A1, A2, B1, B2 = X0
    omega_d = np.sqrt(k/m - (nu/(2*m))**2)
    x = np.exp(-nu/(2*m) * t) * (A1 * np.cos(omega_d * t) + B1 * np.sin(omega_d * t))
    y = np.exp(-nu/(2*m) * t) * (A2 * np.cos(omega_d * t) + B2 * np.sin(omega_d * t))
    pos = np.vstack((x,y)).T
    return pos

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ts = np.linspace(0, 10)
    res = damped_oscillator_analytical(ts, (1, 0), 10, 1)
    plt.plot(ts, res)
    plt.show()