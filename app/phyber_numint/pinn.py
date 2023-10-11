from __future__ import annotations
from typing import Tuple, Optional, Callable, Union
import numpy as np
import warnings
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


    # array_tensor_T = Union[np.ndarray, torch.tensor]
    

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

            self.ts_init: Optional[torch.tensor] = None
            self.xs_init: Optional[torch.tensor] = None
            self.input_init: Optional[torch.tensor] = None
            self.ys_init: Optional[torch.tensor] = None

            self.ts_boundary: Optional[torch.tensor] = None # boundary es contorno
            self.xs_boundary: Optional[torch.tensor] = None
            self.input_boundary: Optional[torch.tensor] = None
            self.ys_boundary: Optional[torch.tensor] = None

            self.ts_physics: Optional[torch.tensor] = None
            self.xs_physics: Optional[torch.tensor] = None
            self.input_physics: Optional[torch.tensor] = None

            self.physics_equation: physics_equation_T = lambda x: x

        def _from_numpy(self, arr: np.ndarray) -> torch.tensor:
            if self.device.type == 'mps':
                if arr.dtype == np.float64:
                    arr = arr.astype(np.float32)
            return torch.from_numpy(arr)
        
        def set_physics_equation(self, equation: physics_equation_T) -> None:
            '''
            equation is a function that takes a torch.tensor as input and outputs a torch.tensor
            The input is the currently calculated values by the pinn. The equation function should
            manipulate these values to shape them in to the physical equation that describes the system
            equated to zero. You can use the gradient, dx and dt member methods to calculate derivatives
            '''
            self.physics_equation = equation

        def set_init_data_xt(self, xs: np.ndarray, ts: np.ndarray, ys: np.ndarray) -> None:
            if not (xs.flatten().shape[0] == ts.flatten().shape[0] == ys.flatten().shape[0]):
                raise TensorShapeError('input shape or output shape are wrong')
            xs = self._from_numpy(xs.flatten())
            ts = self._from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_init = x_grid[:,:,None].requires_grad_(False).to(self.device) # chequear si deberia ser _requires_grad(True)
            self.ts_init = t_grid[:,:,None].requires_grad_(False).to(self.device)
            self.input_init = torch.cat((self.xs_init, self.ts_init), dim=-1).to(self.device)
            self.ys_init = self._from_numpy(ys).view(-1,1).to(self.device)

        def set_init_data_t(self, ts: np.ndarray, ys: np.ndarray) -> None:
            if ts.flatten().shape[0] != ys.flatten().shape[0]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_init = None
            self.ts_init = self._from_numpy(ts)
            self.input_init = self.ts_init.view(-1,1).to(self.device)
            self.ys_init = self._from_numpy(ys).view(-1,1).to(self.device)

        def set_init_data_x(self, xs: np.ndarray, ys: np.ndarray) -> None:
            if xs.flatten().shape[0] != ys.flatten().shape[0]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_init = self._from_numpy(xs)
            self.ts_init = None
            self.input_init = self.xs_init.view(-1,1).to(self.device)
            self.ys_init = self._from_numpy(ys).view(-1,1).to(self.device)

        def set_init_data_free(self, input: np.ndarray, ys: np.ndarray) -> None:
            if input.shape[-1] != self.pinn_sizes[0] or ys.shape[-1] != self.pinn_sizes[-1]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_init = None
            self.ts_init = None
            self.input_init = self._from_numpy(input).to(self.device)
            self.ys_init = self._from_numpy(ys).to(self.device)

        def set_boundary_data_xt(self, xs: np.ndarray, ts: np.ndarray, ys: np.ndarray):
            if not (xs.flatten().shape[0] == ts.flatten().shape[0] == ys.flatten().shape[0]):
                raise TensorShapeError('input shape or output shape are wrong')
            xs = self._from_numpy(xs.flatten())
            ts = self._from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_boundary = x_grid[:,:,None].requires_grad_(False).to(self.device)
            self.ts_boundary = t_grid[:,:,None].requires_grad_(False).to(self.device)
            self.input_boundary = torch.cat((self.xs_boundary, self.ts_boundary), dim=-1).to(self.device)
            self.ys_boundary = self._from_numpy(ys).view(-1,1).to(self.device)

        def set_boundary_data_x(self, xs: np.ndarray, ys: np.ndarray) -> None:
            if xs.flatten().shape[0] != ys.flatten().shape[0]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_boundary = self._from_numpy(xs)
            self.ts_boundary = None
            self.input_boundary = self.xs_boundary.view(-1,1).to(self.device)
            self.ys_boundary = self._from_numpy(ys).view(-1,1).to(self.device)

        def set_boundary_data_t(self, ts: np.ndarray, ys: np.ndarray) -> None:
            if ts.flatten().shape[0] != ys.flatten().shape[0]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_boundary = None
            self.ts_boundary = self._from_numpy(ts)
            self.input_boundary = self.ts_boundary.view(-1,1).to(self.device)
            self.ys_boundary = self._from_numpy(ys).view(-1,1).to(self.device)
        
        def set_boundary_data_free(self, input: np.ndarray, ys: np.ndarray) -> None:
            if input.shape[-1] != self.pinn_sizes[0] or ys.shape[-1] != self.pinn_sizes[-1]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_boundary = None
            self.ts_boundary = None
            self.input_boundary = self._from_numpy(input).to(self.device)
            self.ys_boundary = self._from_numpy(ys).to(self.device)

        def set_physics_data_xt(self, xs: np.ndarray, ts: np.ndarray):
            xs = self._from_numpy(xs.flatten())
            ts = self._from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_physics = x_grid[:,:,None].requires_grad_(True).to(self.device)
            self.ts_physics = t_grid[:,:,None].requires_grad_(True).to(self.device)
            self.input_physics = torch.cat((self.xs_physics, self.ts_physics), dim=-1).to(self.device)

        def set_physics_data_t(self, ts: np.ndarray) -> None:
            self.xs_physics = None
            self.ts_physics = self._from_numpy(ts).requires_grad_(True).to(self.device)
            self.input_physics = self.ts_physics.view(-1,1)

        def set_physics_data_x(self, xs: np.ndarray) -> None:
            self.xs_physics = self._from_numpy(xs).requires_grad_(True).to(self.device)
            self.ts_physics = None
            self.input_physics = self.xs_physics.view(-1,1)

        def set_physics_data_free(self, input: np.ndarray) -> None:
            if input.shape[-1] != self.pinn_sizes[0]:
                raise TensorShapeError('input shape or output shape are wrong')
            self.xs_physics = None
            self.ts_physics = None
            self.input_physics = self._from_numpy(input).to(self.device)

        @staticmethod
        def gradient(y_physics: torch.tensor, var_physics: torch.tensor) -> torch.tensor:
            '''
            calculates d(y_physics)/d(var_physics)
            var_physics should be self.xs_physics or self.ts_physics
            '''
            return torch.autograd.grad(y_physics, var_physics, torch.ones_like(y_physics), create_graph=True)[0]
        
        def dx(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.xs_physics)
        
        def dt(self, y_physics: torch.tensor) -> torch.tensor:
            return self.gradient(y_physics, self.ts_physics)
        
        @staticmethod
        def loss_function(tensor: torch.tensor) -> float:
            return torch.mean(tensor**2)
        
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

        def eval(self, input: Union[np.ndarray, torch.tensor], auto_reshape: bool=True) -> np.ndarray:
            if isinstance(input, np.ndarray):
                input = self._from_numpy(input).to(self.device)
            elif input.device != self.device:
                input.to(self.device)
            result = self.pinn(input if input.shape[-1] == self.pinn_sizes[0] else input.reshape(-1, self.pinn_sizes[0]))
            return result.detach().cpu().numpy().squeeze()
        
        def __call__(self, input: Union[np.ndarray, torch.tensor], auto_reshape: bool=True) -> np.ndarray:
            return self.eval(input, auto_reshape)
        
        def __str__(self) -> str:
            return f'PINN({self.pinn_sizes})'
        
        def __repr__(self) -> str:
            return self.__str__()

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
    pinn.set_init_data_xt(x_init, t_init, y_init)
    pinn.set_boundary_data_xt(x_boundary, t_boundary, y_boundary)
    pinn.set_physics_data_xt(x_physics, t_physics)

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
    # pinn.set_init_data_t(t_data, damped_oscillator_analytical(t_data, X0, k, nu))
    # pinn.set_physics_data_t(t_physics)
    # pinn.train(10_000)

    # res = pinn(t_physics)
    # plt.plot(t_physics, res)
    # plt.plot(t_physics, damped_oscillator_analytical(t_physics, X0, k, nu), ls=':')
    # plt.show()