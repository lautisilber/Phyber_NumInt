import numpy as np
from typing import Tuple, Union

def lorenz_system(t: float, X: Union[Tuple[float,...], np.ndarray], beta: float, sigma: float, rho: float):
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

def damped_oscillator(t: float, X: Union[Tuple[float,...], np.ndarray], k: float, nu: float, m: float=1):
    '''
        mx'' + nu x' + k x = 0

        u' = v
        v' = -(nu/m) v - (k / m) u
    '''
    u, v = X
    u_p = v
    v_p = -(nu / m) * v - (k / m) * u
    return (u_p, v_p)

def damped_oscillator_analytical(t: Union[float, np.ndarray], X0: Tuple[float, float], k: float, nu: float, m: float=1):
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