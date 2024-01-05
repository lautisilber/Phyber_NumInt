from __future__ import annotations
from typing import Tuple, List, Optional, Callable, Union
import numpy as np
import warnings
from numpy import ndarray

import torch
from torch._C import device
try:
    import torch
    TORCH_EXISTS = True
except ImportError:
    TORCH_EXISTS = False
    warnings.warn(
        'torch is not installed. Install it to take full advantage of its features')
try:
    from tqdm.auto import trange
    TQDM_EXISTS = True
except ImportError:
    TQDM_EXISTS = False


physics_equation_t = Union[Callable[[Tuple[torch.tensor]], torch.tensor], Callable[[torch.tensor], torch.tensor]]
array_t = Union[np.ndarray, torch.Tensor]

class TensorShapeError(Exception):
    pass


class PINNLayerSizeError(Exception):
    pass


class PINNGridSizeError(Exception):
    pass


class PINNDimensionError(Exception):
    pass

# def setdiff1d_torch(t1: torch.tensor, t2: torch.tensor) -> torch.tensor:
#     # https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
#     # all values in t2 have to be in t1 as well
#     combined = torch.cat((t1, t2))
#     uniques, counts = combined.unique(return_counts=True)
#     difference = uniques[counts == 1]
#     # intersection = uniques[counts > 1]
#     return difference

if TORCH_EXISTS:
    class MLP(torch.nn.Module):
        '''
            Multilayer perceptron (MLP) // Perceptríon Multicapa .

            Esta clase define una red neuronal feedforward con múltiples capas ocultas
            lineales, funciones de activación tangente hiperbólica en  las capas ocultas
            y una salida lineal.

            Args:
                sizes (lista): Lista de enteros que especifica el número de neuronas en
                cada capa. El primer elemento debe coincidir con la dimensión de entrada
                y el último con la dimensión de salida.

            Atributos:
                capas (torch.nn.ModuleList): Lista que contiene las capas lineales del MLP.

            Métodos:
                forward(x): Realiza una pasada hacia adelante a través de la red MLP.

            Ejemplo:
                tamaños = [entrada_dim, oculta1_dim, oculta2_dim, salida_dim]
                mlp = MLP(tamaños)
                tensor_entrada = torch.tensor([...])
                salida = mlp(tensor_entrada)
        '''

        def __init__(self, sizes):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for i in range(len(sizes)-1):
                self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1]))

        def forward(self, x):
            h = x
            for hidden in self.layers[:-1]:
                h = torch.tanh(hidden(h))
            output = self.layers[-1]
            y = output(h)
            return y

    class PINN:
        def __init__(self, sizes: Tuple[int, ...],
                     initial_conditions_input: Optional[Tuple[np.ndarray]], initial_conditions_output: Optional[np.ndarray],
                     boundary_conditions_input: Optional[Tuple[np.ndarray]], boundary_conditions_output: Optional[np.ndarray],
                     physics_evaluation_input: Optional[Tuple[np.ndarray]],
                     initial_conditions_grad: bool=False, boundary_conditions_grad: bool=False, physics_grad: bool=True,
                     physics_weight: float = 1e-1, learning_rate: float = 1e-4, device: Optional[Union[torch.device, str]] = None) -> None:

            # The '*_input' variables are tuples. Each element of the tuples are arrays with the discretized steps that will be evaluated
            # for each dimension. For example, if the time domain is 1D (t) and the spatial domain is 2D (x, y) and, specifically, the
            # following discretization was made: t = linspace(0, 5, 5), x = linspace(6, 9, 3), y = linspace(10, 12, 2) the following 3D grid
            # will be created as a input: concatenate(t_grid, x_grid, y_grid), with t_grid, x_grid, y_grod = meshgrid(t, x, y, indexing='ij')
            # this allows the differentiation between t, x and y variables for partial derivatives.

            if not(len(initial_conditions_input) == len(initial_conditions_input) == len(physics_evaluation_input)):
                raise PINNDimensionError('Not all inputs are of same dimension')
            self.input_ndim = len(physics_evaluation_input)
            if self.input_ndim != sizes[0]:
                raise PINNDimensionError(f'The input dimension of the network ({sizes[0]}) is not the same as the number of dimensions of the domain ({self.input_ndim})')

            self.pinn_sizes = sizes
            self.physics_weight = physics_weight
            self.pinn_learning_rate = learning_rate

            if isinstance(device, str):
                try:
                    device = torch.device(device)
                except:
                    device = None
            if device is None:
                self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available(
                ) else "mps" if torch.backends.mps.is_available() else "cpu")
            else:
                self.device: torch.device = device

            self.pinn = MLP(self.pinn_sizes).to(self.device)
            self.pinn_optimizer = torch.optim.Adam(
                self.pinn.parameters(), lr=self.pinn_learning_rate)
            
            # the '*_grad' parameters inidcate wether for initial_conditions, boundary_conditions or the physics
            # the gradient needs to be tracked to make derivatives

            # initial conditions
            # these are the inputs and outputs that are going to be evaluated for the input conditions
            self.train_with_init: bool = initial_conditions_input is not None and initial_conditions_output is not None
            if self.train_with_init:
                self.init_input, self.init_grids = self._grid_from_dims(initial_conditions_input, initial_conditions_grad)
                if not initial_conditions_input.shape[-1] == self.pinn_sizes[0]:
                    raise PINNLayerSizeError('initial_conditions_input shape wrong')
                if not initial_conditions_output.shape[-1] == self.pinn_sizes[-1]:
                    raise PINNLayerSizeError('initial_conditions_output shape wrong')
                self.init_output: Optional[torch.tensor] = self._numpy2torch_smart(initial_conditions_output, initial_conditions_grad)
            else:
                self.init_input: Optional[torch.tensor] = None
                self.init_output: Optional[torch.tensor] = None
                self.init_grids: Optional[Tuple[torch.tensor,...]] = None

            # boundary conditions
            # these are the inputs and outputs that are going to be evaluated for the boundary conditions
            self.train_with_boundary: bool = boundary_conditions_input is not None and boundary_conditions_output is not None
            if self.train_with_boundary:
                self.boundary_input, self.boundary_grids = self._grid_from_dims(boundary_conditions_input)
                if not boundary_conditions_input.shape[-1] == self.pinn_sizes[0]:
                    raise PINNLayerSizeError('boundary_conditions_input shape wrong')
                if not boundary_conditions_output.shape[-1] == self.pinn_sizes[-1]:
                    raise PINNLayerSizeError('boundary_conditions_output shape wrong')
                self.boundary_output: Optional[torch.tensor] = self._numpy2torch_smart(boundary_conditions_output, boundary_conditions_grad)
            else:
                self.boundary_input: Optional[torch.tensor] = None
                self.boundary_output: Optional[torch.tensor] = None
                self.boundary_grids: Optional[Tuple[torch.tensor,...]] = None

            # physics
            self.train_with_physics: bool =  physics_evaluation_input is not None
            if self.train_with_physics:
                self.physics_input, self.physics_grids = self._grid_from_dims(boundary_conditions_input, physics_grad)
                if not physics_evaluation_input.shape[-1] == self.pinn_sizes[0]:
                    raise PINNLayerSizeError('physics_evaluation_input shape wrong')
            else:
                self.physics_input: Optional[torch.tensor] = None
                self.physics_grids: Optional[Tuple[torch.tensor,...]] = None

            self.physics_equation: physics_equation_t = lambda x: x

        def add_physics_equation(self, physics_equation: physics_equation_t) -> None:
            '''
            equation is a function that takes a torch.tensor as input and outputs a torch.tensor
            The input is the currently calculated values by the pinn. The equation function should
            manipulate these values to shape them in to the physical equation that describes the system
            equated to zero. You can use the gradient, dx and dt member methods to calculate derivatives
            '''
            self.physics_equation = physics_equation

        def _numpy2torch(self, arr: np.ndarray, requires_grad: bool = False) -> torch.tensor:
            if self.device.type == 'mps':
                if arr.dtype in (np.float64, np.float80, np.float96, np.float128, np.float256):
                    arr = arr.astype(np.float32)
            return torch.tensor(arr, requires_grad=requires_grad)
        
        def _numpy2torch_smart(self, arr: np.ndarray, requires_grad: bool = False) -> torch.tensor:
            if isinstance(arr, np.ndarray):
                return self._numpy2torch(arr, requires_grad).to(self.device)
            if not isinstance(arr, torch.Tensor):
                raise TypeError(f'arr is not of type numpy.ndarray or torch.Tensor. Instead is of type {type(arr)}')
            if input.device != self.device:
                arr = arr.to(self.device)
            return arr
        
        def _grid_from_dims(self, dims: Tuple[np.ndarray,...], requires_grad:bool=False) -> Tuple[torch.Tensor, Tuple[torch.Tensor,...]]:
            '''
            This function takes a tuple of arrays, typically linspaces or aranges. Each array contains
            the information of the discretization along one axis or dimension (the number of dimensions
            is equal to len(dims)). This funcion turns those arrays into a single array, input, that is
            a grid that is the input of the network. It also returns a tuple of arrays that can be used
            to partially differentiate the network along each dimsnion or axis.
            '''
            grids = torch.meshgrid(*(self._numpy2torch_smart(d) for d in dims), indexing='ij')
            grids = tuple(torch.unsqueeze(g, -1).requires_grad_(requires_grad) for g in grids)
            input = torch.cat(grids, dim=-1)
            return input, grids
        
        @staticmethod
        def gradient(y: torch.tensor, var: torch.tensor) -> torch.tensor:
            '''
            calculates d(y)/d(var)
            var should be a slice (or the whole tensor) of self.input_physics
            (or self.input_init or self.input_boundary, if they have requires_grade = True)
            (because these values are used as input for the pinn)
            '''
            return torch.autograd.grad(y, var, torch.ones_like(y), create_graph=True)[0]
        
        def init_axis(self, axis: int) -> torch.Tensor:
            return self.init_grids[axis]
        
        def boundary_axis(self, axis: int) -> torch.Tensor:
            return self.boundary_grids[axis]
        
        def physics_axis(self, axis: int) -> torch.Tensor:
            return self.physics_grids[axis]

        def partial_diff_init(self, output: torch.tensor, axis: int) -> torch.tensor:
            '''
            partially differentiates with respect to the axis or dimension of the axis index (axis) provided, for the initial conditions
            '''
            return self.gradient(output, self.init_axis(axis))
        
        def partial_diff_boundary(self, output: torch.tensor, axis: int) -> torch.tensor:
            '''
            partially differentiates with respect to the axis or dimension of the axis index (axis) provided, for the boundary conditions
            '''
            return self.gradient(output, self.boundary_axis(axis))
        
        def partial_diff_physics(self, output: torch.tensor, axis: int) -> torch.tensor:
            '''
            partially differentiates with respect to the axis or dimension of the axis index (axis) provided, for the physics evaluation points
            '''
            return self.gradient(output, self.physics_axis(axis))
        
        def train(self, iterations: int):
            if TQDM_EXISTS:
                iter = trange(iterations)
            else:
                iter = range(iterations)
            for epoch in iter:
                self.pinn_optimizer.zero_grad()

                # initial conditions
                if self.train_with_init:
                    init_output = self.pinn(self.input_init)
                    loss_init = self.loss_function(init_output-self.init_output)
                else:
                    loss_init = 0

                # boundary conditions
                if self.train_with_boundary:
                    boundary_output = self.pinn(self.input_boundary)
                    loss_boundary = self.loss_function(
                        boundary_output - self.boundary_output)
                else:
                    loss_boundary = 0

                # physics
                if self.train_with_physics:
                    physics_output = self.pinn(self.physics_input)
                    equation = self.physics_equation(physics_output)
                    loss_physics = self.loss_function(equation)
                else:
                    loss_physics = 0

                loss = loss_init + loss_boundary + self.physics_weight * \
                    loss_physics  # Se suma el error de la física con el de los datos
                loss.backward()
                self.pinn_optimizer.step()

                if epoch % 100 == 0:
                    with torch.autograd.no_grad():
                        if TQDM_EXISTS:
                            iter.set_description(
                                f'Loss: {loss.data:.4e}, Loss_init: {loss_init:.3e}, Loss_boundary: {loss_boundary:.3e}, Loss_physics: {loss_physics:.3e}')
                            iter.refresh()
                        else:
                            print(epoch, "- Traning Loss:", loss.data)

            def eval(self, input: Union[np.ndarray, torch.tensor], auto_squeeze: bool=True) -> np.ndarray:
                if not input.shape[-1] == self.pinn_sizes[0]:
                    raise PINNLayerSizeError('input has wring shape')
                input = self._numpy2torch_smart(input)
                result = self.pinn(input)
                result_np = result.detach().cpu().numpy()
                if auto_squeeze:
                    return result_np.squeeze()
                return result_np

            def __call__(self, input: Union[np.ndarray, torch.tensor], auto_squeeze: bool=True) -> np.ndarray:
                return self.eval(input, auto_squeeze)

            def __str__(self) -> str:
                return f'PINN({self.pinn_sizes})'

            def __repr__(self) -> str:
                return self.__str__()

    class PINN_1D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             tuple(initial_conditions_input) if initial_conditions_input is not None else initial_conditions_input, initial_conditions_output,
                             tuple(boundary_conditions_input) if boundary_conditions_input is not None else boundary_conditions_input, boundary_conditions_output,
                             tuple(physics_evaluation_input) if physics_evaluation_input is not None else physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, physics_weight, learning_rate, device)
            '''
            assumes only dimension is time (t)
            '''
        
        def dt(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 0)
        
        def dt_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the initial condition
            '''
            return self.partial_diff_init(output, 0)
        
        def dt_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the boundary condition
            '''
            return self.partial_diff_boundary(output, 0)
        
    class PINN_2D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, physics_weight, learning_rate, device)
            if not self.input_ndim == 2:
                raise PINNDimensionError(f'Domain dimension is not 2 (it is {self.input_ndim})')
            '''
            assumes dimensions (x, t)
            '''
        
        def dx(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to space for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 0)
        
        def dx_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to space for the initial condition
            '''
            return self.partial_diff_init(output, 0)
        
        def dx_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to space for the boundary condition
            '''
            return self.partial_diff_boundary(output, 0)
        
        def dt(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 1)
        
        def dt_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the initial condition
            '''
            return self.partial_diff_init(output, 1)
        
        def dt_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the boundary condition
            '''
            return self.partial_diff_boundary(output, 1)
        
    class PINN_3D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, physics_weight, learning_rate, device)
            if not self.input_ndim == 3:
                raise PINNDimensionError(f'Domain dimension is not 3 (it is {self.input_ndim})')
            '''
            assumes dimensions (x, y, t)
            '''
        
        def dx(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 0)
        
        def dx_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the initial condition
            '''
            return self.partial_diff_init(output, 0)
        
        def dx_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the boundary condition
            '''
            return self.partial_diff_boundary(output, 0)
        
        def dy(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to y for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 1)
        
        def dy_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to y for the initial condition
            '''
            return self.partial_diff_init(output, 1)
        
        def dy_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to y for the boundary condition
            '''
            return self.partial_diff_boundary(output, 1)
        
        def dt(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 2)
        
        def dt_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the initial condition
            '''
            return self.partial_diff_init(output, 2)
        
        def dt_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the boundary condition
            '''
            return self.partial_diff_boundary(output, 2)
        
    class PINN_4D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, physics_weight, learning_rate, device)
            if not self.input_ndim == 4:
                raise PINNDimensionError(f'Domain dimension is not 4 (it is {self.input_ndim})')
            '''
            assumes dimensions (x, y, t)
            '''
        
        def dx(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 0)
        
        def dx_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the initial condition
            '''
            return self.partial_diff_init(output, 0)
        
        def dx_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the boundary condition
            '''
            return self.partial_diff_boundary(output, 0)
        
        def dy(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 1)
        
        def dy_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the initial condition
            '''
            return self.partial_diff_init(output, 1)
        
        def dy_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to x for the boundary condition
            '''
            return self.partial_diff_boundary(output, 1)
        
        def dz(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to z for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 2)
        
        def dz_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to z for the initial condition
            '''
            return self.partial_diff_init(output, 2)
        
        def dz_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to z for the boundary condition
            '''
            return self.partial_diff_boundary(output, 2)
        
        def dt(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the physics evaluation points
            '''
            return self.partial_diff_physics(output, 3)
        
        def dt_init(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the initial condition
            '''
            return self.partial_diff_init(output, 3)
        
        def dt_boundary(self, output: torch.tensor) -> torch.Tensor:
            '''
            derivative with respect to time for the boundary condition
            '''
            return self.partial_diff_boundary(output, 3)


else:
    class NoTorch:
        def __init__(self) -> None:
            raise ModuleNotFoundError("torch was not found!")

    class MLP(NoTorch):
        pass

    class PINN(NoTorch):
        pass

    class PINN_T(NoTorch):
        pass

    class PINN_X(NoTorch):
        pass

    class PINN_XT(NoTorch):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    def damped_oscillator_pinn():
        from examples import damped_oscillator_analytical
        pinn = PINN([1] + [16]*4 + [1], learning_rate=1e-2,
                    physics_weight=5e-1, device='cpu')

        t_init = np.array([0], dtype=np.float32)
        y_init = np.array([1], dtype=np.float32)

        t_physics = np.linspace(0, 10, 1000, dtype=np.float32)

        def equation(y):
            # \ddot{x} - mu\dot{x} - kx = 0
            mu = 1
            k = 10
            dxdt = pinn.gradient(y, pinn.input_physics)
            dxxddt = pinn.gradient(dxdt, pinn.input_physics)
            return dxxddt + mu*dxdt + k*y

        pinn.set_physics_equation(equation)
        pinn.set_init_condition(t_init, y_init)
        pinn.set_physics_condition(t_physics)

        pinn.train(5000)

        res = pinn(t_physics)
        plt.plot(t_physics, res)
        plt.plot(t_physics, damped_oscillator_analytical(
            t_physics, (1, 0), 10, 1))
        plt.show()

    def damped_oscillator_pinn_2d():
        from examples import damped_oscillator_2d_analytical
        pinn = PINN_2D([2] + [16]*4 + [2], learning_rate=1e-2,
                    physics_weight=5e-1, device='cpu')

        t_init = np.array([0], dtype=np.float32)
        y_init = np.array([1, 1], dtype=np.float32)

        t_physics = np.linspace(0, 10, 1000, dtype=np.float32)

        def equation(y):
            # \ddot{x} - mu\dot{x} - kx = 0
            mu = 1
            k = 10
            dxdt = pinn.gradient(y, pinn.input_physics)
            dxxddt = pinn.gradient(dxdt, pinn.input_physics)
            return dxxddt + mu*dxdt + k*y

        pinn.set_physics_equation(equation)
        pinn.set_init_condition(t_init, y_init)
        pinn.set_physics_condition(t_physics)

        pinn.train(5000)

        res = pinn(t_physics)
        plt.plot(t_physics, res)
        plt.plot(t_physics, damped_oscillator_2d_analytical(
            t_physics, (1, 1, 0, 0), 10, 1))
        plt.show()


    def damped_oscillator_pinn_t():
        from examples import damped_oscillator_analytical
        pinn = PINN_T([1] + [16]*4 + [1], learning_rate=1e-2,
                      physics_weight=5e-1, device='cpu')

        t_init = np.array([0], dtype=np.float32)
        y_init = np.array([1], dtype=np.float32)

        t_physics = np.linspace(0, 10, 1000, dtype=np.float32)

        def equation(y):
            # \ddot{x} - mu\dot{x} - kx = 0
            mu = 1
            k = 10
            # dxdt = pinn.dt(y)
            # dxxddt = pinn.dt(dxdt)
            dxdt = pinn.gradient(y, pinn.input_physics)
            dxxddt = pinn.gradient(dxdt, pinn.input_physics)
            return dxxddt + mu*dxdt + k*y

        pinn.set_physics_equation(equation)
        pinn.set_init_condition(t_init, y_init)
        pinn.set_physics_condition(t_physics)

        pinn.train(5000)

        res = pinn(t_physics)
        plt.plot(t_physics, res)
        plt.plot(t_physics, damped_oscillator_analytical(
            t_physics, (1, 0), 10, 1))
        plt.show()

    damped_oscillator_pinn()
    # from examples import damped_oscillator_analytical

    # X0 = (1.0, 0.0)
    # k, nu = (324, 4)
    # t_data = np.linspace(0, 0.5, 20)
    # t_physics = np.linspace(0, 1, 1000)

    # pinn = PINN([1,32,32,32,1], physics_weight=0.1, learning_rate=1e-4)

    # def equation(ys: torch.tensor) -> torch.tensor:
    #     dt = pinn.dt(ys)
    #     dtt = pinn.dt(dt)
    #     return dtt - nu*dt - ys

    # pinn.set_physics_equation(equation)
    # pinn.set_init_condition_t(t_data, damped_oscillator_analytical(t_data, X0, k, nu))
    # pinn.set_physics_condition_t(t_physics)
    # pinn.train(10_000)

    # res = pinn(t_physics)
    # plt.plot(t_physics, res)
    # plt.plot(t_physics, damped_oscillator_analytical(t_physics, X0, k, nu), ls=':')
    # plt.show()
