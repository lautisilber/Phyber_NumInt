from __future__ import annotations
from typing import Tuple, List, Optional, Callable, Union
import numpy as np
import warnings
from os.path import exists

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


physics_equation_t = Union[Callable[[Tuple[torch.tensor]],
                                    torch.tensor], Callable[[torch.tensor], torch.tensor]]
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
                     initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True,
                     name: Optional[str] = None, autoload_model: bool = True,
                     physics_weight: float = 1e-1, learning_rate: float = 1e-4, device: Optional[Union[torch.device, str]] = None) -> None:

            # The '*_input' variables are tuples. Each element of the tuples are arrays with the discretized steps that will be evaluated
            # for each dimension. For example, if the time domain is 1D (t) and the spatial domain is 2D (x, y) and, specifically, the
            # following discretization was made: t = linspace(0, 5, 5), x = linspace(6, 9, 3), y = linspace(10, 12, 2) the following 3D grid
            # will be created as a input: concatenate(t_grid, x_grid, y_grid), with t_grid, x_grid, y_grod = meshgrid(t, x, y, indexing='ij')
            # this allows the differentiation between t, x and y variables for partial derivatives.

            self.input_ndim = len(physics_evaluation_input)
            used_inputs = tuple(i for i in (
                initial_conditions_input, boundary_conditions_input, physics_evaluation_input) if i is not None)
            if not all(len(i) == self.input_ndim for i in used_inputs):
                raise PINNDimensionError(
                    'Not all inputs are of same dimension')
            if self.input_ndim != sizes[0]:
                raise PINNDimensionError(
                    f'The input dimension of the network ({sizes[0]}) is not the same as the number of dimensions of the domain ({self.input_ndim})')

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
            # self.pinn = torch.compile(self.pinn)
            self.pinn_optimizer = torch.optim.Adam(
                self.pinn.parameters(), lr=self.pinn_learning_rate)

            # the '*_grad' parameters inidcate wether for initial_conditions, boundary_conditions or the physics
            # the gradient needs to be tracked to make derivatives

            # initial conditions
            # these are the inputs and outputs that are going to be evaluated for the input conditions
            self.train_with_init: bool = initial_conditions_input is not None and initial_conditions_output is not None
            if self.train_with_init:
                self.init_input, self.init_grids = self._grid_from_dims(
                    initial_conditions_input, initial_conditions_grad)
                if not initial_conditions_output.shape[-1] == self.pinn_sizes[-1]:
                    raise PINNLayerSizeError(
                        'initial_conditions_output shape wrong')
                self.init_output: Optional[torch.tensor] = self._numpy2torch_smart(
                    initial_conditions_output, initial_conditions_grad)
            else:
                self.init_input: Optional[torch.tensor] = None
                self.init_output: Optional[torch.tensor] = None
                self.init_grids: Optional[Tuple[torch.tensor, ...]] = None

            # boundary conditions
            # these are the inputs and outputs that are going to be evaluated for the boundary conditions
            self.train_with_boundary: bool = boundary_conditions_input is not None and boundary_conditions_output is not None
            if self.train_with_boundary:
                self.boundary_input, self.boundary_grids = self._grid_from_dims(
                    boundary_conditions_input)
                if not boundary_conditions_output.shape[-1] == self.pinn_sizes[-1]:
                    raise PINNLayerSizeError(
                        'boundary_conditions_output shape wrong')
                self.boundary_output: Optional[torch.tensor] = self._numpy2torch_smart(
                    boundary_conditions_output, boundary_conditions_grad)
            else:
                self.boundary_input: Optional[torch.tensor] = None
                self.boundary_output: Optional[torch.tensor] = None
                self.boundary_grids: Optional[Tuple[torch.tensor, ...]] = None

            # physics
            self.train_with_physics: bool = physics_evaluation_input is not None
            if self.train_with_physics:
                self.physics_input, self.physics_grids = self._grid_from_dims(
                    physics_evaluation_input, physics_grad)
            else:
                self.physics_input: Optional[torch.tensor] = None
                self.physics_grids: Optional[Tuple[torch.tensor, ...]] = None

            self.physics_equation: physics_equation_t = lambda x: x

            # autoload model if previous version exists
            self.name: Optional[str] = name
            if autoload_model and self.name:
                if exists(self.name+'.pt'):
                    self.load()

        def set_physics_equation(self, physics_equation: physics_equation_t) -> None:
            '''
            equation is a function that takes a torch.tensor as input and outputs a torch.tensor
            The input is the currently calculated values by the pinn. The equation function should
            manipulate these values to shape them in to the physical equation that describes the system
            equated to zero. You can use the gradient, dx and dt member methods to calculate derivatives
            '''
            self.physics_equation = physics_equation

        def _numpy2torch(self, arr: np.ndarray, requires_grad: bool = False) -> torch.tensor:
            if self.device.type == 'mps':
                if arr.dtype in (np.float64,):
                    arr = arr.astype(np.float32)
            return torch.tensor(arr, requires_grad=requires_grad)

        def _numpy2torch_smart(self, arr: np.ndarray, requires_grad: bool = False) -> torch.tensor:
            if isinstance(arr, np.ndarray):
                return self._numpy2torch(arr, requires_grad).to(self.device)
            if not isinstance(arr, torch.Tensor):
                raise TypeError(
                    f'arr is not of type numpy.ndarray or torch.Tensor. Instead is of type {type(arr)}')
            if arr.device != self.device:
                arr = arr.to(self.device)
            return arr

        def _grid_from_dims(self, dims: Tuple[np.ndarray, ...], requires_grad: bool = False) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
            '''
            This function takes a tuple of arrays, typically linspaces or aranges. Each array contains
            the information of the discretization along one axis or dimension (the number of dimensions
            is equal to len(dims)). This funcion turns those arrays into a single array, input, that is
            a grid that is the input of the network. It also returns a tuple of arrays that can be used
            to partially differentiate the network along each dimsnion or axis.
            '''
            grids = torch.meshgrid(*(self._numpy2torch_smart(d)
                                   for d in dims), indexing='ij')
            grids = tuple(torch.unsqueeze(
                g, -1).requires_grad_(requires_grad) for g in grids)
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

        @staticmethod
        def loss_function(tensor: torch.tensor) -> float:
            return torch.mean(tensor**2)  # MSE

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

        def train(self, iterations: int, save_if_exception: bool = True):
            def do_train():
                if TQDM_EXISTS:
                    iter = trange(iterations)
                else:
                    iter = range(iterations)
                for epoch in iter:
                    self.pinn_optimizer.zero_grad()

                    # initial conditions
                    if self.train_with_init:
                        init_output = self.pinn(self.init_input)
                        loss_init = self.loss_function(
                            init_output-self.init_output)
                    else:
                        loss_init = 0

                    # boundary conditions
                    if self.train_with_boundary:
                        boundary_output = self.pinn(self.boundary_input)
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
                                    f'Loss: {loss:.4e}, Loss_init: {loss_init:.3e}, Loss_boundary: {loss_boundary:.3e}, Loss_physics: {loss_physics:.3e}')
                                iter.refresh()
                            else:
                                print(epoch, "- Traning Loss:", loss.data)

            if save_if_exception:
                try:
                    do_train()
                except Exception as err:
                    self.save()
                    print(err)
            else:
                do_train()

        def eval(self, input: Union[Tuple[np.ndarray, ...], Tuple[torch.tensor, ...]], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            if len(input) != self.pinn_sizes[0]:
                raise PINNDimensionError('input has wrong dimension')
            # input_reshaped = tuple(i.reshape(-1, self.pinn_sizes[0]) if auto_resize else i for i in input)
            input_reshaped = tuple(self._numpy2torch_smart(i) for i in input)
            mesh, _ = self._grid_from_dims(input_reshaped)
            result = self.pinn(mesh)
            result_np = result.detach().cpu().numpy()
            if auto_squeeze:
                return result_np.squeeze()
            return result_np

        def __call__(self, input: Union[np.ndarray, torch.tensor], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return self.eval(input, auto_resize, auto_squeeze)

        def __str__(self) -> str:
            return f'PINN({self.pinn_sizes})'

        def __repr__(self) -> str:
            return self.__str__()

        def save(self) -> None:
            # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            if self.name is None:
                raise Exception(
                    'Can\'t save since model has no name. Specify one in the constructor.')
            torch.save({
                'model_state_dict': self.pinn.state_dict(),
                'optimizer_state_dict': self.pinn_optimizer.state_dict()
            }, self.name+'.pt')

        def load(self) -> None:
            # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html
            if self.name is None:
                raise Exception(
                    'Can\'t load since model has no name. Specify one in the constructor.')
            fname = self.name+'.pt'
            if not exists(fname):
                raise FileNotFoundError(f'File {fname} doesn\'t exists.')
            checkpoint = torch.load(fname)
            self.pinn.load_state_dict(checkpoint['model_state_dict'])
            self.pinn_optimizer.load_state_dict(
                checkpoint['optimizer_state_dict'])

    class PINN_1D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, name: Optional[str] = None, autoload_model: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             (initial_conditions_input,
                              ) if initial_conditions_input is not None else initial_conditions_input, initial_conditions_output,
                             (boundary_conditions_input,
                              ) if boundary_conditions_input is not None else boundary_conditions_input, boundary_conditions_output,
                             (physics_evaluation_input,
                              ) if physics_evaluation_input is not None else physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, name, autoload_model, physics_weight, learning_rate, device)
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

        def eval(self, input: Union[np.ndarray, torch.tensor], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return super().eval((input,), auto_resize, auto_squeeze)

    class PINN_2D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, physics_weight: float = 0.1, name: Optional[str] = None, autoload_model: bool = True, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, name, autoload_model, physics_weight, learning_rate, device)
            if not self.input_ndim == 2:
                raise PINNDimensionError(
                    f'Domain dimension is not 2 (it is {self.input_ndim})')
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

        def eval(self, x: Union[np.ndarray, torch.tensor], t: Union[np.ndarray, torch.tensor], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return super().eval((x, t), auto_resize, auto_squeeze)

    class PINN_3D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, name: Optional[str] = None, autoload_model: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, name, autoload_model, physics_weight, learning_rate, device)
            if not self.input_ndim == 3:
                raise PINNDimensionError(
                    f'Domain dimension is not 3 (it is {self.input_ndim})')
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

        def eval(self, x: Union[np.ndarray, torch.tensor], y: Union[np.ndarray, torch.tensor], t: Union[np.ndarray, torch.tensor], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return super().eval((x, y, t), auto_resize, auto_squeeze)

    class PINN_4D(PINN):
        def __init__(self, sizes: Tuple[int, ...], initial_conditions_input: Optional[np.ndarray], initial_conditions_output: Optional[np.ndarray], boundary_conditions_input: Optional[np.ndarray], boundary_conditions_output: Optional[np.ndarray], physics_evaluation_input: Optional[np.ndarray], initial_conditions_grad: bool = False, boundary_conditions_grad: bool = False, physics_grad: bool = True, name: Optional[str] = None, autoload_model: bool = True, physics_weight: float = 0.1, learning_rate: float = 0.0001, device: device | str | None = None) -> None:
            super().__init__(sizes,
                             initial_conditions_input, initial_conditions_output,
                             boundary_conditions_input, boundary_conditions_output,
                             physics_evaluation_input, initial_conditions_grad,
                             boundary_conditions_grad, physics_grad, name, autoload_model, physics_weight, learning_rate, device)
            if not self.input_ndim == 4:
                raise PINNDimensionError(
                    f'Domain dimension is not 4 (it is {self.input_ndim})')
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

        def eval(self, x: Union[np.ndarray, torch.tensor], y: Union[np.ndarray, torch.tensor], z: Union[np.ndarray, torch.tensor], t: Union[np.ndarray, torch.tensor], auto_resize: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return super().eval((x, y, z, t), auto_resize, auto_squeeze)


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

        t_init = np.array([0], dtype=np.float32)
        y_init = np.array([1], dtype=np.float32)
        t_physics = np.linspace(0, 10, 1000, dtype=np.float32)

        pinn = PINN_1D([1] + [16]*4 + [1],
                       t_init, y_init,
                       None, None,
                       t_physics,
                       learning_rate=1e-2,
                       physics_weight=5e-1, device='cpu')

        def equation(y):
            # \ddot{x} - mu\dot{x} - kx = 0
            mu = 1
            k = 10
            # dxdt = pinn.gradient(y, pinn.physics_input)
            # dxxddt = pinn.gradient(dxdt, pinn.physics_input)
            dxdt = pinn.dt(y)
            dxxddt = pinn.dt(dxdt)
            return dxxddt + mu*dxdt + k*y

        pinn.set_physics_equation(equation)

        pinn.train(5000)

        res = pinn(t_physics)
        plt.plot(t_physics, res)
        plt.plot(t_physics, damped_oscillator_analytical(
            t_physics, (1, 0), 10, 1))
        plt.show()

    def burgers_equation_pinn():
        import matplotlib as mpl

        x = np.linspace(0, 2*np.pi, 100, endpoint=True, dtype=np.float32)

        t_init = np.array([0], dtype=np.float32)
        y_init = np.sin(x, dtype=np.float32).reshape(-1, 1)

        t = np.linspace(0, 1, 100, dtype=np.float32)
        x_bounday = np.array([0, 2*np.pi], dtype=np.float32)
        y_boundary = np.array([0], dtype=np.float32)

        pinn = PINN_2D([2] + [20]*16 + [1],
                       (x, t_init), y_init,
                       (x_bounday, t), y_boundary,
                       (x, t),
                       name='burgers1D',
                       learning_rate=10,
                       physics_weight=1e-2, device='cpu')

        def equation(y):
            # du/dt + u*du/dx - nu*ddu/dxx
            nu = 1e-2
            dudt = pinn.dt(y)
            dudx = pinn.dx(y)
            ddudxx = pinn.dx(dudx)
            return dudt + (y * dudx) - (nu * ddudxx)

        pinn.set_physics_equation(equation)

        pinn.train(1000)
        pinn.save()

        res = pinn(x, t)
        print(res.shape)
        plt.plot(x, y_init, color='k')
        for i in np.linspace(0, res.shape[1], 10, endpoint=False, dtype=int):
            plt.plot(x, res[:, i], color=mpl.colormaps['plasma']
                     (i/res.shape[1]))
        plt.show()

    burgers_equation_pinn()
