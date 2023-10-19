from __future__ import annotations
from typing import Tuple, Optional, Callable, Union
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


class TensorShapeError(Exception):
    pass

class PINNLayerSizeError(Exception):
    pass

class PINNGridSizeError(Exception):
    pass


def np_x_t_to_xt(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x_grid, t_grid = np.meshgrid(x, t, indexing='ij')
    return np.concatenate((x_grid, t_grid), axis=None).reshape((x.shape[0], t.shape[0], 2))

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

        def _numpy2torch(self, arr: np.ndarray) -> torch.tensor:
            if self.device.type == 'mps':
                if arr.dtype in (np.float64, np.float80, np.float96, np.float128, np.float256):
                    arr = arr.astype(np.float32)
            return torch.from_numpy(arr)
        
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
        
        def _get_ys_compatible(self, ys: np.ndarray) -> np.ndarray:
            if ys.shape[-1] == self.pinn_sizes[-1]:
                ys_reshape = ys
            else:
                ys_reshape = ys.reshape(-1, self.pinn_sizes[-1])
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

        def set_init_condition(self, input: np.ndarray, ys: np.ndarray) -> None:
            input_init, ys_init = self._get_input_ys_compatible(input, ys)
            if not self._check_input_ys_compatible(input_init, ys_init):
                raise TensorShapeError('input shape or output shapes are wrong')
            self.input_init = self._numpy2torch(input_init).to(self.device)
            self.ys_init = self._numpy2torch(ys_init).to(self.device)

        def set_boundary_condition(self, input: np.ndarray, ys: np.ndarray) -> None:
            input_boundary, ys_boundary = self._get_input_ys_compatible(input, ys)
            if not self._check_input_ys_compatible(input_boundary, ys_boundary):
                raise TensorShapeError('input shape or output shapes are wrong')
            self.input_boundary = self._numpy2torch(input_boundary).to(self.device)
            self.ys_boundary = self._numpy2torch(ys_boundary).to(self.device)

        def set_physics_condition(self, input: np.ndarray) -> None:
            input_physics = self._get_input_compatible(input)
            self.input_physics = self._numpy2torch(input_physics).requires_grad_(True).to(self.device)

        @staticmethod
        def gradient(y_physics: torch.tensor, var_physics: torch.tensor) -> torch.tensor:
            '''
            calculates d(y_physics)/d(var_physics)
            var_physics should be self.xs_physics or self.ts_physics
            '''
            return torch.autograd.grad(y_physics, var_physics, torch.ones_like(y_physics), create_graph=True)[0]
        
        @staticmethod
        def loss_function(tensor: torch.tensor) -> float:
            return torch.mean(tensor**2) # MSE
        
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
            # self.ts_init: Optional[np.ndarray] = None
            # self.ys_init: Optional[np.ndarray] = None
            # self.ts_boundary: Optional[np.ndarray] = None
            # self.ys_boundary: Optional[np.ndarray] = None
            # self.ts_physics: Optional[np.ndarray] = None

        def dt(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.input_physics)

        def __str__(self) -> str:
            return f'PINN_T({self.pinn_sizes})'
        
    class PINN_XT(PINN):
        def __init__(self, sizes: Tuple[int, ...], physics_weight: float = 0.1, learning_rate: float = 0.0001, device: Optional[Union[torch.device, str]] = None) -> None:
            if sizes[0] < 2:
                raise PINNLayerSizeError(f'XT grid based PINN should have a input layer of dimension 12 or greater, not {sizes[0]}')
            super().__init__(sizes, physics_weight, learning_rate, device)
            self.t_size = 1
            self.x_size = sizes[0] - 1
            # self.ts_init: Optional[np.ndarray] = None
            # self.ys_init: Optional[np.ndarray] = None
            # self.ts_boundary: Optional[np.ndarray] = None
            # self.ys_boundary: Optional[np.ndarray] = None
            self.ts_physics: Optional[torch.tensor] = None
            self.xs_physics: Optional[Tuple[torch.tensor]] = None

        @staticmethod
        def _xs_ts_to_xt_grid(xs: np.ndarray, ts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            ts = ts.flatten()
            xs = xs.reshape(-1, xs.shape[-1])
            mesh_shape = (*([xs.shape[0]] * xs.shape[-1]), ts.shape[0], xs.shape[-1] + 1)
            # Create arrays for x, y, and t
            m = np.array(np.meshgrid(*[xs[:, i] for i in range(xs.shape[-1])], ts)) # each dimension on each of the firs axis' indices
            # Stack them along the last axis to form the mesh
            mesh = np.stack(m, axis=-1).reshape(mesh_shape)
            for i in range(xs.shape[-1] // 2):
                mesh = np.swapaxes(mesh, i, xs.shape[-1]-i-1)
            return mesh, m
        
        def _get_input_compatible(self, xs: np.ndarray, ts: np.ndarray) -> np.ndarray:
            mesh, _ = self._xs_ts_to_xt_grid(xs, ts)
            return mesh
        
        def _get_input_compatible_torch(self, xs: torch.tensor, ts: torch.tensor) -> torch.tensor:
            xs_np = xs.detach().numpy()
            ts_np = ts.detach().numpy()
            mesh = self._get_input_compatible(xs_np, ts_np)
            return self._numpy2torch(mesh).to(self.device)
        
        def _get_input_ys_compatible(self, xs: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            input_reshape = self._get_input_compatible(xs, ts)
            ys_reshape = self._get_ys_compatible(ys)
            return input_reshape, ys_reshape
        
        def _check_input_ys_compatible(self, xs: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> bool:
            input_reshape, ys_reshape = self._get_input_ys_compatible(xs, ts, ys)
            return (
                np.sum(input_reshape.shape[:-1]) == np.sum(ys_reshape.shape[:-1])
            )

        def set_init_condition(self, xs: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> None:
            input = self._get_input_compatible(xs, ts)
            super().set_init_condition(input, ys)

        def set_physics_condition(self, xs: np.ndarray, ts:np.ndarray) -> None:
            input, m = self._xs_ts_to_xt_grid(xs, ts)
            self.xs_physics = self._numpy2torch(np.expand_dims(m[:-1], axis=-1)).require_grad_(True).to(self.device)
            self.ts_physics = self._numpy2torch(np.expand_dims(m[-1], axis=-1)).require_grad_(True).to(self.device)
            return super().set_physics_condition(input)

        def dx(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.xs_physics[0])
        
        def dy(self, y_physics: torch.tensor) -> torch.tensor:
            if self.x_size < 2:
                raise PINNGridSizeError(f'Y dimension does not exist in system with {self.x_size} spatial dimentions')
            return self.gradient(y_physics, self.xs_physics[1])
        
        def dz(self, y_physics: torch.tensor) -> torch.tensor:
            if self.x_size < 3:
                raise PINNGridSizeError(f'Z dimension does not exist in system with {self.x_size} spatial dimentions')
            return self.gradient(y_physics, self.xs_physics[2])
        
        def d_spatial(self, y_physics: torch.tensor, spatial_index: int) -> torch.tensor:
            if spatial_index >= self.x_size:
                raise PINNGridSizeError(f'The dimension of index {spatial_index} does not exist in system with {self.x_size} spatial dimentions')
            return self.gradient(y_physics, self.xs_physics[spatial_index])

        def dt(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.ts_physics)
        
        def laplacian(self, y_physics: torch.tensor) -> torch.tensor:
            dt = self.dt(y_physics)
            dt2 = self.dt(dt)
            lap = dt2
            for i in range(self.x_size):
                di = self.d_spatial(y_physics, i)
                di2 = self.d_spatial(di, i)
                lap += di2
            # is this the same as (don't think so due to tensor sizes)
            # dX = self.gradient(y_physics, self.xs_physics)
            # dX2 = self.gradient(dX, self.xs_physics)
            # lap += dX2
            return lap
        
        def eval(self, xs: np.ndarray, ts: np.ndarray, auto_reshape: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            input = self._get_input_compatible(xs, ts)
            return super().eval(input, auto_reshape, auto_squeeze)
        
        def __call__(self, xs: np.ndarray, ts: np.ndarray, auto_reshape: bool = True, auto_squeeze: bool = True) -> np.ndarray:
            return self.eval(xs, ts, auto_reshape, auto_squeeze)

        def __str__(self) -> str:
            return f'PINN_T({self.pinn_sizes})'

else:
    class NoTorch:
        def __init__(self) -> None:
            raise ModuleNotFoundError("torch was not found!")

    class MLP(NoTorch):
        pass

    class PINN(NoTorch):
        pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    Nx = Nt = 100
    tf = 2

    t_init = np.zeros(Nt)
    x_init = np.linspace(0, 2*np.pi, Nx)
    y_init = np.sin(x_init)

    t_boundary = np.concatenate((np.linspace(0,tf,Nt), np.linspace(0,tf,Nt)), axis=0)
    x_boundary = 2*np.pi*np.concatenate((np.zeros(Nx), np.ones(Nx)), axis=0)
    y_boundary = np.concatenate((np.zeros(Nx), np.zeros(Nt)), axis=0)

    t_physics = np.linspace(0, tf, Nt)
    x_physics = np.linspace(0, 2*np.pi, Nx)

    pinn = PINN([2] + ([20]*8) + [1], learning_rate=1e-5, physics_weight=1e-1)

    def equation(ys: torch.tensor) -> torch.tensor:
        dt = pinn.dt(ys)
        dx = pinn.dx(ys)
        return dt + ys * dx
    
    pinn.set_physics_equation(equation)
    pinn.set_init_condition_xt(x_init, t_init, y_init)
    pinn.set_boundary_condition_xt(x_boundary, t_boundary, y_boundary)
    pinn.set_physics_condition_xt(x_physics, t_physics)

    pinn.train(10)

    input = np_x_t_to_xt(x_physics, t_physics)
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