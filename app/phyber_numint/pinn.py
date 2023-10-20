from __future__ import annotations
from typing import Tuple, List, Optional, Callable, Union
import numpy as np
import warnings

import torch
try:
    import torch
    TORCH_EXISTS = True
except ImportError:
    TORCH_EXISTS = False
    warnings.warn('torch is not installed. Install it to take full advantage of its features')
try:
    from tqdm.auto import trange, tqdm
    TQDM_EXISTS = True
except ImportError:
    TQDM_EXISTS = False


physics_equation_T = Callable[[torch.tensor], torch.tensor]
y_func_T = Union[Callable[[np.ndarray],np.ndarray], Callable[[float],float]]


class TensorShapeError(Exception):
    pass

class PINNLayerSizeError(Exception):
    pass

class PINNGridSizeError(Exception):
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
        def __init__(self,sizes):
            super().__init__()
            self.layers = torch.nn.ModuleList()
            for i in range(len(sizes)-1):
                self.layers.append(torch.nn.Linear(sizes[i],sizes[i+1]))
        def forward(self,x):
            h = x
            for hidden in self.layers[:-1]:
                h = torch.tanh(hidden(h))
            output = self.layers[-1]
            y = output(h)
            return y


    
    class PINN:
        def __init__(self, sizes: Tuple[int,...], physics_weight: float=1e-1, learning_rate: float=1e-4, device: Optional[Union[torch.device, str]]=None) -> None:
            self.pinn_sizes = sizes
            self.physics_weight = physics_weight
            self.pinn_learning_rate = learning_rate

            self.ts_training_physics: Optional[torch.tensor] = None

            if isinstance(device, str):
                try:
                    device = torch.device(device)
                except:
                    device = None
            if device is None:
                self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            else:
                self.device: torch.device = device

            self.pinn = MLP(self.pinn_sizes).to(self.device)
            self.pinn_optimizer = torch.optim.Adam(self.pinn.parameters(), lr=self.pinn_learning_rate)

            self.input_init: Optional[torch.tensor] = None
            self.ys_init: Optional[torch.tensor] = None # ys work as output

            self.input_boundary: Optional[torch.tensor] = None
            self.ys_boundary: Optional[torch.tensor] = None

            self.input_physics: Optional[torch.tensor] = None

            self.physics_equation: physics_equation_T = lambda x: x

        def _numpy2torch(self, arr: np.ndarray, requires_grad: bool=False) -> torch.tensor:
            if self.device.type == 'mps':
                if arr.dtype in (np.float64, np.float80, np.float96, np.float128, np.float256):
                    arr = arr.astype(np.float32)
            return torch.tensor(arr, requires_grad=requires_grad)
        
        def _get_input_compatible(self, input: np.ndarray) -> np.ndarray:
            if input.shape[-1] == self.pinn_sizes[0]:
                input_reshape = input
            else:
                input_reshape = input.reshape(-1, self.pinn_sizes[0])
                print('Had to reshape input')
            return input_reshape
        
        def _get_input_compatible_torch(self, input: torch.tensor) -> torch.tensor:
            if input.shape[-1] == self.pinn_sizes[0]:
                input_reshape = input
            else:
                input_reshape = input.view(-1, self.pinn_sizes[0])
                print('Had to reshape input')
            return input_reshape
        
        def _get_ys_compatible(self, ys: np.ndarray, verbose: bool=True) -> np.ndarray:
            if ys.shape[-1] == self.pinn_sizes[-1]:
                ys_reshape = ys
            else:
                ys_reshape = ys.reshape(-1, self.pinn_sizes[-1])
                if verbose:
                    print('Had to reshape ys')
            return ys_reshape
        
        def _get_input_ys_compatible(self, input: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            input_reshape = self._get_input_compatible(input)
            ys_reshape = self._get_ys_compatible(ys)
            return input_reshape, ys_reshape

        def _check_input_ys_compatible(self, input: np.ndarray, ys: np.ndarray) -> bool:
            input_reshape, ys_reshape = self._get_input_ys_compatible(input, ys)
            return (
                np.sum(input_reshape.shape[:-1]) == np.sum(ys_reshape.shape[:-1])
            )
        
        def set_physics_equation(self, equation: physics_equation_T) -> None:
            '''
            equation is a function that takes a torch.tensor as input and outputs a torch.tensor
            The input is the currently calculated values by the pinn. The equation function should
            manipulate these values to shape them in to the physical equation that describes the system
            equated to zero. You can use the gradient, dx and dt member methods to calculate derivatives
            '''
            self.physics_equation = equation

        def set_init_condition(self, input: np.ndarray, ys: np.ndarray, requires_grad: bool=False) -> None:
            input_init, ys_init = self._get_input_ys_compatible(input, ys)
            if not self._check_input_ys_compatible(input_init, ys_init):
                raise TensorShapeError('input shape or output shapes are wrong')
            self.input_init = self._numpy2torch(input_init, requires_grad).to(self.device)
            self.ys_init = self._numpy2torch(ys_init).to(self.device)

        def set_init_condition_constant(self, input: np.ndarray, y: float, requires_grad: bool=False) -> None:
            input_init = self._get_input_compatible(input)
            ys = np.empty((*input_init.shape[:-1], self.pinn_sizes[-1])).fill(y)
            self.set_init_condition(input, ys, requires_grad)

        def set_boundary_condition(self, input: np.ndarray, ys: np.ndarray, requires_grad: bool=False) -> None:
            input_boundary, ys_boundary = self._get_input_ys_compatible(input, ys)
            if not self._check_input_ys_compatible(input_boundary, ys_boundary):
                raise TensorShapeError('input shape or output shapes are wrong')
            self.input_boundary = self._numpy2torch(input_boundary, requires_grad).to(self.device)
            self.ys_boundary = self._numpy2torch(ys_boundary).to(self.device)

        def set_boundary_condition_constant(self, input: np.ndarray, y: float, requires_grad: bool=False) -> None:
            input_boundary = self._get_input_compatible(input)
            ys = np.empty((*input_boundary.shape[:-1], self.pinn_sizes[-1])).fill(y)
            self.input_boundary(input, ys, requires_grad)

        def set_physics_condition(self, input: np.ndarray, requires_grad: bool=True) -> None:
            input_physics = self._get_input_compatible(input)
            self.input_physics = self._numpy2torch(input_physics, requires_grad).to(self.device)

        @staticmethod
        def gradient(y: torch.tensor, var: torch.tensor) -> torch.tensor:
            '''
            calculates d(y)/d(var)
            var should be a slice (or the whole tensor) of self.input_physics
            (or self.input_init or self.input_boundary, if they have requires_grade = True)
            (because these values are used as input for the pinn)
            '''
            return torch.autograd.grad(y, var, torch.ones_like(y), create_graph=True)[0]
        
        # def _partial_derivative(self, y: torch.tensor, var: torch.tensor, axis: Optional[Union[int, torch.tensor]]=None) -> torch.tensor:
        #     '''
        #     Calculates the partial derivative of y with respect to the axis indicated. The shape of the resulting tensor is the same as self.graident
        #     but zeroes out the axes that do not correspond to the axis that was differentiated against
        #     '''
        #     if not y.requires_grad or not var.requires_grad:
        #         raise Exception('Not all tensors require grad. set requires_grad to True when creating them.')
        #     grad = self.gradient(y, var)
        #     if axis is None:
        #         return grad
        #     if isinstance(axis, int):
        #         axis = torch.tensor((axis,), dtype=torch.int)
        #     indices_to_nullify = setdiff1d_torch(torch.arange(var.shape[-1], dtype=torch.int), axis)
        #     return grad[...,indices_to_nullify]
        
        @staticmethod
        def loss_function(tensor: torch.tensor) -> float:
            return torch.mean(tensor**2) # MSE
        
        def train_step(self, input: torch.tensor, optimal: torch.tensor) -> None:
            if input.shape[-1] != self.pinn_sizes[0] or optimal.shape[-1] != self.pinn_sizes[-1]:
                raise PINNLayerSizeError('input or optimal are wrong shape')
            self.pinn_optimizer.zero_grad()
            res = self.pinn(input)
            loss = self.loss_function(res - optimal)
            loss.backward()
            self.pinn_optimizer.step()

        def train(self, iterations: int):
            if TQDM_EXISTS:
                iter = trange(iterations)
            else:
                iter = range(iterations)
            for epoch in iter:
                self.pinn_optimizer.zero_grad()

                # initial conditions
                if self.input_init is not None:
                    y_init = self.pinn(self.input_init)
                    loss_init = self.loss_function(y_init-self.ys_init)
                else:
                    loss_init = 0
                
                # boundary conditions
                if self.input_boundary is not None:
                    y_boundary = self.pinn(self.input_boundary)
                    loss_boundary = self.loss_function(y_boundary - self.ys_boundary)
                else:
                    loss_boundary = 0
                
                # physics
                if self.input_physics is not None:
                    y_physics = self.pinn(self.input_physics)
                    equation = self.physics_equation(y_physics)
                    loss_physics = self.loss_function(equation)
                else:
                    loss_physics = 0

                loss = loss_init + loss_boundary + self.physics_weight * loss_physics #Se suma el error de la física con el de los datos
                loss.backward()
                self.pinn_optimizer.step()

                if epoch % 100 == 0:
                    with torch.autograd.no_grad():
                        if TQDM_EXISTS:
                            iter.set_description(f'Loss: {loss.data:.4e}, Loss_init: {loss_init:.3e}, Loss_boundary: {loss_boundary:.3e}, Loss_physics: {loss_physics:.3e}')
                            iter.refresh()
                        else:
                            print(epoch,"- Traning Loss:",loss.data)

        def eval(self, input: Union[np.ndarray, torch.tensor], auto_reshape: bool=True, auto_squeeze: bool=True) -> np.ndarray:
            if isinstance(input, np.ndarray):
                input_compatible = self._get_input_compatible(input) if auto_reshape else input
                input_compatible = self._numpy2torch(input_compatible).to(self.device)
            elif input.device != self.device:
                input_compatible = (self._get_input_compatible_torch(input) if auto_reshape else input).to(self.device)
            result = self.pinn(input_compatible)
            if auto_squeeze:
                return result.detach().cpu().numpy().squeeze()
            return result.detach().cpu().numpy()
        
        def __call__(self, input: Union[np.ndarray, torch.tensor], auto_reshape: bool=True, auto_squeeze: bool=True) -> np.ndarray:
            return self.eval(input, auto_reshape, auto_squeeze)
        
        def __str__(self) -> str:
            return f'PINN({self.pinn_sizes})'
        
        def __repr__(self) -> str:
            return self.__str__()
        
    class PINN_T(PINN):
        def __init__(self, sizes: Tuple[int, ...], physics_weight: float = 0.1, learning_rate: float = 0.0001, device: Optional[Union[torch.device, str]] = None) -> None:
            if sizes[0] != 1:
                raise PINNLayerSizeError(f'Time based PINN should have a input layer of dimension 1, not {sizes[0]}')
            super().__init__(sizes, physics_weight, learning_rate, device)

        def set_init_condition(self, ts: np.ndarray, ys: np.ndarray, requires_grad: bool=False) -> None:
            return super().set_init_condition(ts, ys, requires_grad)
        
        def set_boundary_condition(self, ts: np.ndarray, ys: np.ndarray, requires_grad: bool=False) -> None:
            return super().set_boundary_condition(ts, ys, requires_grad)
        
        def set_physics_condition(self, ts: np.ndarray, requires_grad: bool=True) -> None:
            return super().set_physics_condition(ts, requires_grad)

        def dt(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.input_physics)

        def __str__(self) -> str:
            return f'PINN_T({self.pinn_sizes})'
        
    class PINN_X(PINN):
        def __init__(self, sizes: Tuple[int, ...], physics_weight: float = 0.1, learning_rate: float = 0.0001, device: Optional[Union[torch.device, str]] = None) -> None:
            super().__init__(sizes, physics_weight, learning_rate, device)

            self.x_grids_init: Optional[Tuple[torch.tensor]] = None # we have to store this to be able to differentiate later
            self.x_grids_boundary: Optional[Tuple[torch.tensor]] = None # we have to store this to be able to differentiate later
            self.x_grids_physics: Optional[Tuple[torch.tensor]] = None # we have to store this to be able to differentiate later

        def _xs_to_x_grid_torch(self, *xss, requires_grad: bool=True) -> Tuple[torch.tensor, Tuple[torch.tensor]]:
            '''
            x_mesh is the mesh which serves as input
            x_grid contains at each index the grid of points relevant to make a partial derivative with respect to the variable of that index
            (that correspondance may be x->0, y->1, z->2, and so on)
            '''
            x_grids = tuple(torch.unsqueeze(x_grid, x_grid.ndim).requires_grad_(requires_grad).to(self.device) for x_grid in torch.meshgrid(*xss, indexing='ij'))
            x_mesh = torch.cat(x_grids, dim=-1)
            return x_mesh, x_grids
        
        def set_init_condition(self, ys: np.ndarray, *xss: np.ndarray, requires_grad: bool=False) -> None:
            if any(xs.ndim != 1 for xs in xss) or len(xss) != self.pinn_sizes[0] or ys.shape[-1] != self.pinn_sizes[-1] or np.sum(np.sum(xs.shape[:-1]) for xs in xss) != np.sum(ys.shape[:-1]):
                raise PINNGridSizeError('Shapes are not right')
            x_mesh, x_grids = self._xs_to_x_grid_torch(*xss, requires_grad)
            self.x_grids_init = x_grids
            self.input_init = x_mesh
        
        def set_boundary_condition(self, ys: np.ndarray, *xss: np.ndarray, requires_grad: bool=False) -> None:
            if any(xs.ndim != 1 for xs in xss) or len(xss) != self.pinn_sizes[0] or ys.shape[-1] != self.pinn_sizes[-1] or np.sum(np.sum(xs.shape[:-1]) for xs in xss) != np.sum(ys.shape[:-1]):
                raise PINNGridSizeError('Shapes are not right')
            x_mesh, x_grids = self._xs_to_x_grid_torch(*xss, requires_grad)
            self.x_grids_boundary = x_grids
            self.input_boundary = x_mesh

        def set_physics_condition(self, *xss: np.ndarray, requires_grad: bool=True) -> None:
            if any(xs.ndim != 1 for xs in xss) or len(xss) != self.pinn_sizes[0]:
                raise PINNGridSizeError('Shapes are not right')
            x_mesh, x_grids = self._xs_to_x_grid_torch(*xss, requires_grad)
            self.x_grids_physics = x_grids
            self.input_physics = x_mesh

        def diff(self, y: torch.tensor, var: Tuple[torch.tensor], axis: int) -> torch.tensor:
            '''
            differentiates with respect to the variable of the axis provided, that is in var
            '''
            return self.gradient(y, var[axis])
        
        def diff_physics(self, y_physics: torch.tensor, axis: int) -> torch.tensor:
            return self.diff(y_physics, self.x_grids_physics, axis)
        
        def diff_init(self, y_init: torch.tensor, axis: int) -> torch.tensor:
            return self.diff(y_init, self.x_grids_init, axis)

        def diff_boundary(self, y_boundary: torch.tensor, axis: int) -> torch.tensor:
            return self.diff(y_boundary, self.x_grids_boundary, axis)
        
        def dx(self, y_physics: torch.tensor) -> torch.tensor:
            return self.diff_physics(y_physics, 0)
        
        def dy(self, y_physics: torch.tensor) -> torch.tensor:
            return self.diff_physics(y_physics, 1)
        
        def dz(self, y_physics: torch.tensor) -> torch.tensor:
            return self.diff_physics(y_physics, 2)
        
        def __str__(self) -> str:
            return f'PINN_X({self.pinn_sizes})'

        
    class PINN_XT(PINN_X):
        def __init__(self, sizes: Tuple[int, ...], physics_weight: float = 0.1, learning_rate: float = 0.0001, device: Optional[Union[torch.device, str]] = None) -> None:
            if sizes[0] < 2:
                raise PINNLayerSizeError(f'XT grid based PINN should have a input layer of dimension 2 or greater, not {sizes[0]}')
            super().__init__(sizes, physics_weight, learning_rate, device)

        def set_init_condition(self, ys: np.ndarray, ts:np.ndarray, *xss: np.ndarray, requires_grad: bool=False) -> None:
            return super().set_init_condition(ys, *xss, ts, requires_grad=requires_grad)

        def set_boundary_condition(self, ys: np.ndarray, ts:np.ndarray, *xss: np.ndarray, requires_grad: bool=False) -> None:
            return super().set_boundary_condition(ys, *xss, ts, requires_grad=requires_grad)
        
        def set_physics_condition(self, ts:np.ndarray, *xss: np.ndarray, requires_grad: bool=True) -> None:
            return super().set_physics_condition(*xss, ts, requires_grad=requires_grad)
        
        def dt(self, y_physics: torch.tensor) -> torch.tensor:
            return self.diff_physics(y_physics, self.pinn_sizes[0]-1)
        
        def __str__(self) -> str:
            return f'PINN_XT({self.pinn_sizes})'

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
        pinn = PINN([1] + [16]*4 + [1], learning_rate=1e-2, physics_weight=5e-1, device='cpu')

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
        plt.plot(t_physics, damped_oscillator_analytical(t_physics, (1, 0), 10, 1))
        plt.show()

    damped_oscillator_pinn()
    def damped_oscillator_pinn_t():
        from examples import damped_oscillator_analytical
        pinn = PINN_T([1] + [16]*4 + [1], learning_rate=1e-2, physics_weight=5e-1, device='cpu')

        t_init = np.array([0], dtype=np.float32)
        y_init = np.array([1], dtype=np.float32)

        t_physics = np.linspace(0, 10, 1000, dtype=np.float32)

        def equation(y):
            # \ddot{x} - mu\dot{x} - kx = 0
            mu = 1
            k = 10
            dxdt = pinn.dt(y)
            dxxddt = pinn.dt(dxdt)
            return dxxddt + mu*dxdt + k*y

        pinn.set_physics_equation(equation)
        pinn.set_init_condition(t_init, y_init)
        pinn.set_physics_condition(t_physics)

        pinn.train(5000)

        res = pinn(t_physics)
        plt.plot(t_physics, res)
        plt.plot(t_physics, damped_oscillator_analytical(t_physics, (1, 0), 10, 1))
        plt.show()

    def pinn_xt():
        Nx = Nt = 100
        tf = 2

        pinn = PINN_XT([2] + ([20]*8) + [1], learning_rate=1e-5, physics_weight=1e-1, device='cpu')
        
        t_init = np.zeros(Nt)
        x_init = np.linspace(0, 2*np.pi, Nx)
        xt_init = pinn.create_grid(x_init, t_init)

        t_boundary = np.concatenate((np.linspace(0,tf,Nt), np.linspace(0,tf,Nt)), axis=0)
        x_boundary = 2*np.pi*np.concatenate((np.zeros(Nx), np.ones(Nx)), axis=0)
        y_boundary = np.concatenate((np.zeros(Nx), np.zeros(Nt)), axis=0)

        t_physics = np.linspace(0, tf, Nt)
        x_physics = np.linspace(0, 2*np.pi, Nx)

        def equation(ys: torch.tensor) -> torch.tensor:
            dt = pinn.dt(ys)
            dx = pinn.dx(ys)
            return dt + ys * dx
        
        pinn.set_physics_equation(equation)
        pinn.set_init_condition_func(x_init, t_init, np.sin, axis=0)
        pinn.set_boundary_condition(x_boundary, t_boundary, y_boundary)
        pinn.set_physics_condition(x_physics, t_physics)

        pinn.train(10)

        input = pinn._get_input_compatible(x_physics, t_physics)
        res = pinn(input)
        plt.close('all')
        plt.imshow(res)
        plt.show()


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