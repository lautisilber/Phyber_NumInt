from __future__ import annotations
from typing import Tuple, Optional, Callable
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
        def __init__(self, sizes: Tuple[int,...], physics_weight: float=1e-1, lr: float=1e-4, device: Optional[torch.device]=None) -> None:
            self.pinn_sizes = sizes
            self.physics_weight = physics_weight
            self.pinn_lr = lr

            self.ts_training_physics: Optional[torch.tensor] = None

            self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") if device is None else device

            self.pinn = MLP(self.sizes).to(device)
            self.pinn_optimizer = torch.optim.Adam(self.pinn.parameters(), lr=self.lr)

            self.ts_init: Optional[torch.tensor] = None
            self.xs_init: Optional[torch.tensor] = None
            self.input_xt_init: Optional[torch.tensor] = None
            self.ys_init: Optional[torch.tensor] = None

            self.ts_boundary: Optional[torch.tensor] = None # boundary es contorno
            self.xs_boundary: Optional[torch.tensor] = None
            self.input_xt_boundary: Optional[torch.tensor] = None
            self.ys_boundary: Optional[torch.tensor] = None

            self.ts_physics: Optional[torch.tensor] = None
            self.xs_physics: Optional[torch.tensor] = None
            self.input_xt_physics: Optional[torch.tensor] = None

            self.physics_equation: physics_equation_T = lambda x: x

        def set_init_data(self, xs: np.ndarray, ts: np.ndarray):
            xs = torch.from_numpy(xs.flatten())
            ts = torch.from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_init = torch.from_numpy(x_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.ts_init = torch.from_numpy(t_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.input_xt_physics= torch.cat((self.xs_init, self.ts_init), dim=-1).to(self.device)

        def set_init_data(self, xs: np.ndarray, ts: np.ndarray):
            xs = torch.from_numpy(xs.flatten())
            ts = torch.from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_boundary = torch.from_numpy(x_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.ts_boundary = torch.from_numpy(t_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.input_xt_physics= torch.cat((self.xs_boundary, self.ts_boundary), dim=-1).to(self.device)

        def set_training_physics(self, xs: np.ndarray, ts: np.ndarray):
            xs = torch.from_numpy(xs.flatten())
            ts = torch.from_numpy(ts.flatten())
            x_grid, t_grid = torch.meshgrid(xs, ts, indexing='ij')
            self.xs_physics = torch.from_numpy(x_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.ts_physics = torch.from_numpy(t_grid)[:,:,None].requires_grad_(True).to(self.device)
            self.input_xt_physics= torch.cat((self.xs_physics, self.ts_physics), dim=-1).to(self.device)

        @staticmethod
        def gradient(y_physics: torch.tensor, var_physics: torch.tensor) -> torch.tensor:
            '''
            calculates d(y_physics)/d(var_physics)
            var_physics should be self.xs_physics or self.ts_physics
            '''
            return torch.autograd.grad(y_physics, var_physics, torch.ones_like(y_physics), create_graph=True)
        
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
                if self.input_xt_init is not None:
                    y_init = self.pinn(self.input_xt_init)
                    loss_init = self.loss_function(y_init-self.ys_init)
                else:
                    loss_init = 0
                
                # boundary conditions
                if self.input_xt_boundary is not None:
                    y_boundary = self.pinn(self.input_xt_boundary)
                    loss_boundary = self.loss_function(y_boundary - self.ys_boundary)
                else:
                    loss_boundary = 0
                
                # physics
                if self.input_xt_physics is not None:
                    y_physics = self.pinn(self.input_xt_physics)
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
                            iter.set_description(f'Loss: {loss.data:.4e}, loss1: {loss_init:.3e}, loss2: {loss_boundary:.3e}, loss3: {loss_physics:.3e}')
                            iter.refresh()
                        else:
                            print(epoch,"- Traning Loss:",loss.data)
else:
    class NoTorch:
        def __init__(self) -> None:
            raise ModuleNotFoundError("torch was not found!")

    class MLP(NoTorch):
        pass

    class PINN(NoTorch):
        pass


if __name__ == '__mian__':
    pinn = PINN()