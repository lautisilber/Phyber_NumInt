# Phyber NumInt

## A numerical integrator for solving ODE systems

A simple package to solving ODE with different algorithms, written as explicitly as possible. I created this for a University course.

To use it you simply create an instance of the class ODEIntegrator

It takes the following parameters

- ```F``` that's the function that represents the ODE system (X' = F(X))
- ```X0``` that's the initial conditions
- ```ti``` the initial time
- ```tf``` the final time
- ```dt``` the time step

The function ```F``` should be structured in the following fashion

```python
def F_lorenz(t, X, sigma, beta, rho):
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
```

Then, the method ```solve``` of ```ODEIntegrator``` can be called with the ```method``` argument. The ```method``` argument is an enum member of IntegrationMethods. The following exist

- ```EULER_FORWARD```
- ```EULER_BACKWARD```
- ```EULER_IMPROVED```
- ```EULER_IMPROVED```
- ```RUNGE_KUTTA_2```
- ```RUNGE_KUTTA_3```
- ```RUNGE_KUTTA_4```
- ```RUNGE_KUTTA_FEHLBERG_45```

For example, ```ode_integrator.solve(IntegrationMethod.EULER_FORWARD)```

Examples in the file *examples.py*