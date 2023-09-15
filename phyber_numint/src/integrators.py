from __future__ import annotations
import numpy as np
from enum import Enum, auto
from typing import Callable, Tuple, Any, Iterable, Dict, Optional, Union#, TypeAlias
import warnings
try:
    from tqdm.auto import trange, tqdm
    TQDM_EXISTS = True
except ImportError:
    TQDM_EXISTS = False
    warnings.warn('tqdm is not installed. Install it to take full advantage of its features')
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_EXISTS = True
except ImportError:
    MATPLOTLIB_EXISTS = False
    warnings.warn('matplotlib is not installed. Install it to take full advantage of its features')


class IntegrationMethod(Enum):
    NONE = auto()
    EULER_FORWARD = auto()
    EULER_BACKWARD = auto()
    EULER_IMPROVED = auto()
    RUNGE_KUTTA_2 = auto()
    RUNGE_KUTTA_3 = auto()
    RUNGE_KUTTA_4 = auto()
    RUNGE_KUTTA_FEHLBERG_45 = auto()

    def __str__(self) -> str:
        return ' '.join(self.name.split('_')).title()

#F_type: TypeAlias = Callable[[float, Tuple[float, ...], Any], Tuple[float, ...]]
F_type = Callable[[float, Tuple[float, ...], Any], Tuple[float, ...]]

def _get_iterator(ts, desc, use) -> Iterable:
    '''
        ts is an iterable
        desc is a str
        use is a bool
    '''
    if TQDM_EXISTS and use:
        return trange(len(ts), desc=desc)
    else:
        return range(len(ts))
    
def _broyden_method(f: Callable[[Tuple[float, ...]], np.ndarray], x0: Tuple[float, ...], tol: float=1e-6, max_iter: float=100) -> np.ndarray:
    '''
        Find the root of a vector-valued function using Broyden's method.

        Parameters:
        - f: The vector-valued function for which you want to find the root.
        - x0: Initial guess for the root as a NumPy array.
        - tol: Tolerance for the root approximation (default is 1e-6).
        - max_iter: Maximum number of iterations (default is 100).

        Returns:
        - root: Approximation of the root as a NumPy array.
        - iterations: Number of iterations taken.

        https://en.wikipedia.org/wiki/Broyden%27s_method
    '''

    n = len(x0)
    x = x0
    B = np.eye(n)  # Initial approximation of the Jacobian (identity matrix)
    iterations = 0

    while np.linalg.norm(f(x)) > tol and iterations < max_iter:
        delta_x = np.linalg.solve(B, -f(x))
        x_new = x + delta_x
        delta_f = f(x_new) - f(x)

        # Broyden's update formula for the Jacobian approximation
        B += np.outer(delta_f - np.dot(B, delta_x), delta_x) / np.dot(delta_x, delta_x)

        x = x_new
        iterations += 1

    return x

def _euler_forward(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, show_tqdm: bool):
    '''
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the time step

        https://en.wikipedia.org/wiki/Euler_method

        x_n+1 = x_n + h*F(t_n, x_n)
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Euler', show_tqdm)
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    Xs[0,:] = X0
    for i in iter:
        Xs[i+1,:] = Xs[i,:] + dt*np.array(F(ts[i], Xs[i,:]))
    return ts, Xs

def _euler_back(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, error_tolerance: float, show_tqdm: bool):
    '''
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the time step
        error_tolerance is a float and is the error tolerance for the boyden method for root finding (default is 1e-6).

        https://en.wikipedia.org/wiki/Backward_Euler_method

        x_n+1 = x_n + h*F(t_n+1, x_n+1)
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Euler (back)', show_tqdm)
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    Xs[0:2,:] = X0
    for i in iter:
        Xs[i+1,:] = _broyden_method(lambda X_ip1: Xs[i,:] + dt*F(ts[i+1], X_ip1) - X_ip1, Xs[i,:], error_tolerance)
    return ts, Xs

def _euler_improved(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, show_tqdm: bool):
    '''
    F is the differntial Function
    X0 is a 1d array with initial conditions (has to be the right shape)
    ti is a float and is the initial time
    tf is a float and is the final time
    dt is a float and is the time step

    https://flexbooks.ck12.org/cbook/ck-12-calculus-concepts/section/8.14/primary/lesson/numerical-methods-for-solving-odes-calc/
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Euler Improved (Heun)', show_tqdm)
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    Xs[0:2,:] = X0
    for i in iter:
        s = F(ts[i], Xs[i,:])
        Xs[i+1,:] = Xs[i,:] + .5*dt*( s + F(ts[i+1], Xs[i,:] + dt*s) )
    return ts, Xs

def _rk2(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, show_tqdm: bool):
    '''
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the time step

        https://www.mathstools.com/section/main/Metodos_de_Runge_Kutta?lang=en
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Runge Kutta 2', show_tqdm)
    h = dt
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    Xs[0,:] = X0
    for i in iter:
        k1 = F(ts[i], Xs[i,:])
        k2 = F(ts[i+1], Xs[i,:] + k1)
        Xs[i+1] = Xs[i] + .5 * h * (k1 + k2)
    return ts, Xs

def _rk3(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, show_tqdm: bool):
    '''
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the time step

        https://flexbooks.ck12.org/cbook/ck-12-calculus-concepts/section/8.14/primary/lesson/numerical-methods-for-solving-odes-calc/
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Runge Kutta 3', show_tqdm)
    h = dt
    h_half = 0.5 * h
    one_sixth = 1/6
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    Xs[0,:] = X0
    for i in iter:
        k1 = F(ts[i], Xs[i,:])
        k2 = F(ts[i] + h_half, Xs[i,:] + k2 * h_half)
        k3 = F(ts[i+1], Xs[i,:] - k1*h + 2*k2*h)
        Xs[i+1] = Xs[i] + one_sixth * h * (k1 + 4 * k2 + k3)
    return ts, Xs

def _rk4(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, show_tqdm: bool):
    '''
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the time step

        https://www.mathstools.com/section/main/Metodos_de_Runge_Kutta?lang=en
    '''
    ts = np.arange(ti, tf, dt)
    iter = _get_iterator(ts[1:], 'Runge Kutta 4', show_tqdm)
    h = dt
    Xs = np.zeros((len(ts), len(X0)), dtype=float)
    h_half = .5 * h
    one_sixth = 1/6
    Xs[0,:] = X0
    for i in iter:
        k1 = F(ts[i], Xs[i,:])
        k2 = F(ts[i] + h_half, Xs[i,:] + h_half * k1)
        k3 = F(ts[i] + h_half, Xs[i,:] + h_half * k2)
        k4 = F(ts[i+1], Xs[i,:] + h * k3)
        Xs[i+1] = Xs[i] + h * one_sixth * (k1 + 2*k2 + 2*k3 + k4)
    return ts, Xs

def _rkf45(F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, error_tolerance: float, show_tqdm: bool):
    '''
        ts is an array of times
        F is the differntial Function
        X0 is a 1d array with initial conditions (has to be the right shape)
        ti is a float and is the initial time
        tf is a float and is the final time
        dt is a float and is the initial time step
        error_tolerance is a float and is the error tolerance for adaptive step size (default is 1e-6).

        https://en.wikipedia.org/wiki/Runge–Kutta–Fehlberg_method
    '''

    ts = []
    Xs = []
    h = dt

    t = ti
    X = X0

    def _rkf_45_step():
        nonlocal ts, Xs, h, t, X
        ts.append(t)
        Xs.append(X)

        # Calculate the 5th and 4th order Runge-Kutta steps
        k1 = h * F(t, X)
        k2 = h * F(t + h/5, X + k1/5)
        k3 = h * F(t + 3*h/10, X + 3*k1/40 + 9*k2/40)
        k4 = h * F(t + 3*h/5, X + 3*k1/10 - 9*k2/10 + 6*k3/5)
        k5 = h * F(t + h, X - 11*k1/54 + 5*k2/2 - 70*k3/27 + 35*k4/27)
        k6 = h * F(t + 7*h/8, X + 1631*k1/55296 + 175*k2/512 + 575*k3/13824 + 44275*k4/110592 + 253*k5/4096)

        # Calculate the error estimate
        error = np.max(np.abs(1/360 * k1 - 128/4275 * k3 - 2197/75240 * k4 + 1/50 * k5 + 2/55 * k6))

        # Calculate the adaptive step size
        h_new = 0.9 * h * (error_tolerance / error)**0.2

        # Update the step size for the next iteration
        h = min(h_new, tf - t)

        # Update time and y for the next iteration
        t += h
        X = X + 25/216 * k1 + 1408/2565 * k3 + 2197/4104 * k4 - 1/5 * k5

    if TQDM_EXISTS and show_tqdm:
        with tqdm(total=(tf-ti), desc='RKF 45') as p_bar:
            while t < tf:
                _rkf_45_step()
                p_bar.update(h)
    else:
        while t < tf:
            _rkf_45_step()

    return np.array(ts), np.array(Xs)

class NoIntegrationMethodException(Exception):
    pass

class ODEIntegrator:
    '''
        This integrator can solve ODEs with the folloewing structure:
        X'(t) = F(X(t))
        Where X is a vector of variables that are dependant on t and
        F is an arbitrary function that describes the differential equation
        (or system of differential equations) and that takes the X vector and
        produces a new vector with the same shape.
    '''
    def __init__(self, F: F_type, X0: Tuple[float, ...], ti: float, tf: float, dt: float, integration_method: IntegrationMethod=IntegrationMethod.NONE, error_tolerance: float=1e-6, F_args: Optional[Tuple[Any]]=None, F_kwargs: Optional[Dict[str, Any]]=None) -> None:
        self.original_F: F_type = F
        self.F_args = F_args if F_args is not None else tuple()
        self.F_kwargs = F_kwargs if F_kwargs is not None else dict()
        self.F: F_type = self.original_F
        self.set_F_params(self.F_args, self.F_kwargs)
        self.X0 = X0
        self.ti = ti
        self.tf = tf
        self.dt = dt
        self.integration_method = integration_method
        self.error_tolerance = error_tolerance # for adaptative time step methods

        self.ts: Optional[np.ndarray] = None
        self.Xs: Optional[np.ndarray] = None

    def set_F_params(self, F_args: Optional[Tuple[Any]]=None, F_kwargs: Optional[Dict[str, Any]]=None) -> None:
        self.F_args = F_args if F_args is not None else tuple()
        self.F_kwargs = F_kwargs if F_kwargs is not None else dict()
        self.F: F_type = lambda *args, **kwargs: np.array(self.original_F(*args, *F_args, **kwargs, **F_kwargs))

    def __str__(self) -> str:
        return str(self.integration_method)
    
    def set_integration_method(self, integration_method: IntegrationMethod):
        if not type(integration_method) is IntegrationMethod:
            raise TypeError(f'integration_error should be of type {IntegrationMethod} and is of type {type(integration_method)}')
        self.integration_method = integration_method

    def solve(self, integration_method: Optional[IntegrationMethod]=None, show_tqdm: bool=True) -> tuple[np.ndarray, np.ndarray]:
        if integration_method is not None:
            self.set_integration_method(integration_method)

        if self.integration_method is IntegrationMethod.NONE:
            raise NoIntegrationMethodException('No integration method was specified')
        elif self.integration_method is IntegrationMethod.EULER_FORWARD:
            ts, Xs = _euler_forward(self.F, self.X0, self.ti, self.tf, self.dt, show_tqdm)
        elif self.integration_method is IntegrationMethod.EULER_BACKWARD:
            ts, Xs = _euler_back(self.F, self.X0, self.ti, self.tf, self.dt, self.error_tolerance, show_tqdm)
        elif self.integration_method is IntegrationMethod.EULER_IMPROVED:
            ts, Xs = _euler_improved(self.F, self.X0, self.ti, self.tf, self.dt, show_tqdm)
        elif self.integration_method is IntegrationMethod.RUNGE_KUTTA_2:
            ts, Xs = _rk2(self.F, self.X0, self.ti, self.tf, self.dt, show_tqdm)
        elif self.integration_method is IntegrationMethod.RUNGE_KUTTA_3:
            ts, Xs = _rk3(self.F, self.X0, self.ti, self.tf, self.dt, show_tqdm)
        elif self.integration_method is IntegrationMethod.RUNGE_KUTTA_4:
            ts, Xs = _rk4(self.F, self.X0, self.ti, self.tf, self.dt, show_tqdm)
        elif self.integration_method is IntegrationMethod.RUNGE_KUTTA_FEHLBERG_45:
            ts, Xs = _rkf45(self.F, self.X0, self.ti, self.tf, self.dt, self.error_tolerance, show_tqdm)
        else:
            raise Exception
        self.ts = ts
        self.Xs = Xs
        return self.ts, self.Xs

    def show(self, axes_to_show: Union[Tuple[int, ...], bool], with_time: str='none', ts: Optional[np.ndarray]=None, Xs: Optional[np.ndarray]=None, ax=None, plt_args: Optional[Tuple[Any, ...]]=None, plt_kwargs: Optional[Dict[str, Any]]=None, plt_show: bool=False) -> Union[None, Any]:
        '''
            - axes_to_show is a tuple with the index numbers of the solution that will be shown in the plot. if true, all
              will be shown. if false (don't do this) nothing happens
            - with_time is 'axis' if a time axis should be added. 'phase' to show a phase diagram and colour
              the line as a heatmap to signify time progression. 'none' omits time
            - ts is the time array. It is None if the stored should be used
            - Xs is the solution array. It is None if the stored should be used
            - ax is the axis to which the plot should be drawn. It is None if a new axis should be created
            - plt_args are args that can be passed to the plt.plot function
            - plt_kwargs are kwargs that can be passed to the plt.plot function
            - plt_show calls plt.show() if true
        '''

        if not MATPLOTLIB_EXISTS:
            return
        
        plt_args = plt_args if plt_args is not None else tuple()
        plt_kwargs = plt_kwargs if plt_kwargs is not None else dict()
        
        with_time = with_time.lower()
        if with_time not in ('axis', 'phase', 'none'):
            raise Exception(f"The argument 'wiht_time' should be one of the following options: ('axis', 'phase', 'none'), but is {with_time}.")
        
        ts = ts if ts is not None else self.ts
        Xs = Xs if Xs is not None else self.Xs
        
        if type(axes_to_show) is bool:
            if axes_to_show:
                Xs_to_show = Xs
            else:
                return
        else:
            Xs_to_show = Xs[:,tuple(axes_to_show)]

        def _gen_ax(is_3d: bool):
            nonlocal ax
            if ax is None:
                if is_3d:
                    ax = plt.subplot(111, projection='3d')
                else:
                    ax = plt.subplot(111)

        if with_time == 'none':
            if Xs_to_show.shape[1] > 3:
                if Xs_to_show.shape[1] > 4:
                    raise Exception(f'Too many axes to show. Max is 4, trying to show {Xs_to_show.shape[1]}')
                _gen_ax(True)
                line = ax.plot(*[Xs_to_show[:,i] for i in range(Xs_to_show.shape[1])][:3], c=Xs_to_show[:,3], *plt_args, **plt_kwargs)
            else:
                _gen_ax(Xs_to_show.shape[1] == 3)
                line = ax.plot(*[Xs_to_show[:,i] for i in range(Xs_to_show.shape[1])], *plt_args, **plt_kwargs)
        elif with_time == 'axis':
            if Xs_to_show.shape[1] > 2:
                raise Exception(f'Too many axes to show. Max is 2, trying to show {Xs_to_show.shape[1]}')
            _gen_ax(Xs_to_show.shape[1] == 2)
            line = ax.plot(*[Xs_to_show[:,i] for i in range(Xs_to_show.shape[1])], ts, *plt_args, **plt_kwargs)
        elif with_time == 'phase':
            if Xs_to_show.shape[1] > 3:
                raise Exception(f'Too many axes to show. Max is 3, trying to show {Xs_to_show.shape[1]}')
            _gen_ax(Xs_to_show.shape[1] == 3)
            line = ax.scatter(*[Xs_to_show[:,i] for i in range(Xs_to_show.shape[1])], c=ts, *plt_args, **plt_kwargs)
        else:
            raise Exception
        
        if plt_show:
            plt.show()
        else:
            return ax, line