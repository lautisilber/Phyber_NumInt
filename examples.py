from phyber_numint import ODEIntegrator, IntegrationMethod
from phyber_numint import lorenz_system, damped_oscillator, damped_oscillator_analytical
import matplotlib.pyplot as plt

def lorenz():
    integ = ODEIntegrator(lorenz_system, (0., .5, .5), 0, 50, 1e-4, F_args=(8/3, 10, 25))
    ts, Xs = integ.solve(IntegrationMethod.RUNGE_KUTTA_4)

    plt.subplot(111, projection='3d')
    plt.plot(Xs[:,0], Xs[:,1], Xs[:,2])
    plt.title(str(integ))
    plt.show()

def lorenz2():
    integ = ODEIntegrator(lorenz_system, (0., .5, .5), 0, 50, 1e-3, F_args=(8/3, 10, 25))
    integ.solve(IntegrationMethod.RUNGE_KUTTA_4)
    integ.show(True, plt_show=True)
    integ.show((0,1), 'phase', plt_show=True, plt_kwargs={'s': 2})


def damped_osc():
    X0 = (1., 0.)
    k, nu = 1., .5
    integ = ODEIntegrator(damped_oscillator, X0, 0, 25, 1e-4, F_args=(k, nu))
    ts_back, Xs_back = integ.solve(IntegrationMethod.EULER_BACKWARD)
    ts_forward, Xs_forward = integ.solve(IntegrationMethod.EULER_FORWARD)
    ts_45, Xs_45 = integ.solve(IntegrationMethod.RUNGE_KUTTA_FEHLBERG_45)
    Xs_analytical = damped_oscillator_analytical(ts_45, X0, k, nu)

    plt.plot(ts_back, Xs_back[:,0], '-o', markersize=2, label=str(IntegrationMethod.EULER_BACKWARD))
    plt.plot(ts_forward, Xs_forward[:,0], '-o', markersize=2, label=str(IntegrationMethod.EULER_FORWARD))
    plt.plot(ts_45, Xs_45[:,0], '-o', markersize=2, label=str(IntegrationMethod.RUNGE_KUTTA_FEHLBERG_45))
    plt.plot(ts_45, Xs_analytical, '-o', markersize=2, label='Analytical')
    plt.legend()
    plt.show()

def damped_osc_2():
    integ = ODEIntegrator(damped_oscillator, (1, 0), 0, 25, 1e-3, F_args=(1, .5))
    integ.solve(IntegrationMethod.RUNGE_KUTTA_FEHLBERG_45)
    integ.show(True, 'phase', plt_show=True)

if __name__ == '__main__':
    lorenz()
    lorenz2()
    damped_osc()
    damped_osc_2()