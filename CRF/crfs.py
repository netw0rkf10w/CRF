"""Copyright 2020 D. Khue Le-Huu
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import pydensecrf.densecrf as dcrf

from .PermutohedralFiltering.PermutohedralFiltering import PermutohedralLayer
from .convcrf import *
from .sparsemax import sparsemax

import matplotlib.pyplot as plt
def savemat(x, path):
    plt.imshow(x, vmin=0, vmax=1)
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    plt.savefig(path, dpi=600, bbox_inches='tight')

def unnormalize(tensor, mean, std, inplace=False):
    """Unnormalize a tensor image with mean and standard deviation.

    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.

    See :class:`~torchvision.transforms.Normalize` for more details.

    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.

    Returns:
        Tensor: Normalized Tensor image.
    """
    if not torch.is_tensor(tensor):
        raise TypeError('tensor should be a torch tensor. Got {}.'.format(type(tensor)))

    if tensor.ndimension() != 3:
        raise ValueError('Expected tensor to be a tensor image of size (C, H, W). Got tensor.size() = '
                         '{}.'.format(tensor.size()))

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if (std == 0).any():
        raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
    if mean.ndim == 1:
        mean = mean[:, None, None]
    if std.ndim == 1:
        std = std[:, None, None]
    tensor.mul_(std).add_(mean)
    return tensor


class ADMMParams():
    """ADMM parameter class

    Args:
        init (str, Optional): 'zeros', 'uniform', 'softmax', 'unary', 'unary_solution'
    """
    def __init__(self, rho=1.0, rho_type='scalar', projection='bregman',
                 loose=False, init='softmax'):
    
        self.rho = rho
        self.rho_type = rho_type
        self.projection = projection
        self.loose = loose
        self.init = init

    def __str__(self):
        return (f"ADMMParams: \n\t rho:\t\t {self.rho} \n\t rho_type:\t {self.rho_type} \n\t init:\t\t {self.init} \n\t "
                + f"projection:\t {self.projection} \n\t decomposition:\t {'loose' if self.loose else 'strict'}")


class FrankWolfeParams():
    """Frank-Wolfe parameter class

    Args:
        stepsize (float, Optional): stepsize. Only applicable for the 'fixed' scheme.
        scheme (str, Optional): 'fixed', 'standard', 'linesearch'
    """
    def __init__(self, stepsize=1.0, scheme='linesearch', regularizer=None, lambda_=1.0, lambda_learnable=False, x0_weight=0.0, x0_weight_learnable=False):
        self.stepsize = stepsize
        self.scheme = scheme
        assert scheme in ['fixed', 'standard', 'linesearch']
        self.regularizer = regularizer
        assert regularizer in [None, 'negentropy', 'l2']
        self.lambda_ = lambda_
        self.lambda_learnable = lambda_learnable
        self.x0_weight = x0_weight
        self.x0_weight_learnable = x0_weight_learnable

    def __str__(self):
        return (f"FrankWolfeParams: \n\t scheme:\t {self.scheme} \n\t "
                + f"stepsize:\t {self.stepsize} (for the 'fixed' scheme) \n\t "
                + f"regularizer:\t {self.regularizer}\n\t "
                + f"lambda_:\t {self.lambda_}\n\t "
                + f"lambda_learnable:\t {self.lambda_learnable}\n\t "
                + f"x0_weight:\t {self.x0_weight}\n\t "
                + f"x0_weight_learnable:\t {self.x0_weight_learnable}")


class GeneralCRF(nn.Module):
    r"""Base class for all CRFs
    """
    
    def __init__(self, solver='mf', iterations=5, params=None,
                 ergodic=False, x0_weight=0.0, output_logits=True, print_energy=False):

        super().__init__()
        self.print_energy = print_energy
        self.iterations = iterations
        # self.classes = classes
        self.solver = solver
        self.output_logits = output_logits
        self.ergodic = ergodic
        self.x0_weight = x0_weight
        self.params = params

        print(f'CRF solver: {solver}')
        print(f'x0_weight: {x0_weight}')

        if solver == 'admm':
            if params is not None:
                assert (type(params) is ADMMParams)
            else:
                print('Use default ADMM parameters')
                self.params = ADMMParams()
            
            print(self.params)

            self.loose = params.loose
            self.projection = params.projection

            rho = params.rho
            rho_type = params.rho_type
            if rho_type == 'fixed':
                self.rho = rho
            elif rho_type == 'scalar':
                self.rho = torch.nn.Parameter(torch.tensor(rho))
                # self.rho_scale = torch.nn.Parameter(torch.tensor(admm_rho))
            elif rho_type == 'vector':
                raise NotImplementedError
            elif rho_type == 'tensor':
                # self.rho_conv = nn.Conv2d(classes, classes, kernel_size=(1, 1))
                # nn.init.zeros_(self.rho_conv.weight)
                # nn.init.zeros_(self.rho_conv.bias)
                # self.rho = None
                # self.rho = torch.nn.Parameter(torch.zeros(classes) + admm_rho)
                raise NotImplementedError
            else:
                raise NotImplementedError
        elif solver == 'fw':
            if params is not None:
                assert (type(params) is FrankWolfeParams)
            else:
                print('Use default Frank-Wolfe parameters')
                self.params = FrankWolfeParams()
            print(self.params)

            if self.params.lambda_learnable:
                self.fw_lambda = torch.nn.Parameter(torch.tensor(self.params.lambda_))
                print(f'Trainable lambda for Frank-Wolfe. Initialized at {self.params.lambda_}')
            else:
                self.fw_lambda = self.params.lambda_
                print(f'Non-trainable lambda for Frank-Wolfe: {self.params.lambda_}')

            if self.params.x0_weight_learnable:
                self.fw_x0_weight = torch.nn.Parameter(torch.tensor(self.params.x0_weight))
                print(f'Trainable x0_weight for Frank-Wolfe. Initialized at {self.params.x0_weight}')
            else:
                self.fw_x0_weight = self.params.x0_weight
                print(f'Non-trainable x0_weight for Frank-Wolfe: {self.params.x0_weight}')


    def compute_pairwise(self, x, image):
        """Compute P * x
        This function is specific to the derived classes.
        """
        raise NotImplementedError

    def energy(self, x, image, unaries, pairwise=None):
        """Sum of energies over the batch
        """
        if pairwise is None:
            pairwise = self.compute_pairwise(x, image)
        return torch.sum(x*(0.5*pairwise - unaries))

    def energy_discrete(self, x, image, unaries):
        one_hot = torch.argmax(x, dim=1, keepdim=True)
        one_hot = torch.zeros(x.shape, device=x.device).scatter_(1, one_hot, 1)
        return self.energy(one_hot, image, unaries)

    def forward(self, image, unaries, return_energy=False):
        """
        Shape:
            image: (B, channels, H, W)
            unaries: (B, classes, H, W)
        """
        if self.solver == 'mf':
            return self.mean_field(image, unaries, return_energy=return_energy)
        elif self.solver == 'admm':
            return self.admm(image, unaries, return_energy=return_energy)
        elif self.solver == 'fw':
            return self.frank_wolfe(image, unaries, return_energy=return_energy)
        elif self.solver == 'pgd':
            return self.pgd(image, unaries, return_energy=return_energy)
        elif self.solver == 'proxgrad':
            return self.proximal_gradient(image, unaries, return_energy=return_energy)
        else:
            raise NotImplementedError

    def mean_field(self, image, unaries, return_energy=False):
        
        # Debug: unnormalize image
        # image = unnormalize(image[0], mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        # image = image*255
        # image = image.unsqueeze(0)

        if self.iterations < 1:
            return unaries

        x0_weight = self.x0_weight
        x_weight = 1.0 - x0_weight

        # set the q_values
        q_values = unaries
        s = 0
        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)
        for i in range(self.iterations):
            q_values = F.softmax(q_values, dim=1)
            if i == 0 and x0_weight > 0:
                x0 = q_values

            if self.ergodic:
                s = s + q_values

            pairwise = self.compute_pairwise(q_values, image)

            if self.print_energy or return_energy:
                if not self.ergodic:
                    e = self.energy(q_values, image, unaries, pairwise=pairwise)
                    e_discrete = self.energy_discrete(q_values, image, unaries)
                else:
                    e = self.energy(s/(i+1), image, unaries)
                    e_discrete = self.energy_discrete(s/(i+1), image, unaries)
                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # 4. add pairwise terms
            q_values = unaries - pairwise

        if self.ergodic:
            s = (s + F.softmax(q_values, dim=1))/(1+self.iterations)
            if self.print_energy or return_energy:
                e = self.energy(s, image, unaries)
                e_discrete = self.energy_discrete(s, image, unaries)
                energies[i+1] = e
                energies_discrete[i+1] = e_discrete
                if self.print_energy:
                    print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')
            if self.output_logits:
                s = torch.log(s + 1e-9)
            q_values = s
        else:
            if self.print_energy or return_energy:
                q_normalized = F.softmax(q_values, dim=1)
                e = self.energy(q_normalized, image, unaries)
                e_discrete = self.energy_discrete(q_normalized, image, unaries)
                energies[i+1] = e
                energies_discrete[i+1] = e_discrete
                if self.print_energy:
                    print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')
            if not self.output_logits:
                q_values = F.softmax(q_values, dim=1)
                if x0_weight > 0:
                    q_values = x_weight*q_values + x0_weight*x0
            else:
                if x0_weight > 0:
                    # print(f'x0_weight: {x0_weight}')
                    q_values = F.softmax(q_values, dim=1)
                    q_values = x_weight*q_values + x0_weight*x0
                    q_values = torch.log(q_values + 1e-9)
        
        if return_energy:
            return q_values, energies, energies_discrete
        return q_values
    

    def pgd(self, image, unaries, return_energy=False):
        """Projected gradient descent
        """

        if self.iterations < 1:
            return unaries

        x0_weight = self.x0_weight
        x_weight = 1.0 - x0_weight
        # initialization
        x0 = F.softmax(unaries, dim=1)
        x = x0
        Px = self.compute_pairwise(x, image)

        # set the q_values
        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)
        for i in range(self.iterations):
            
            if self.print_energy or return_energy:
                e = self.energy(x, image, unaries, pairwise=Px)
                e_discrete = self.energy_discrete(x, image, unaries)
                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # Project x - grad onto the constraint set
            s = x - Px + unaries
            s = sparsemax(s, 1)

            # Update x = x + alpha*(s - x)
            linesearch = True
            s = s - x # r = s - x
            if not linesearch:
                # Basic step: alpha = 2/(k+2)
                alpha = 2.0/(2.0 + i)
            else:
                Pr = self.compute_pairwise(s, image)
                A = torch.sum(s*Pr, dim=[1, 2, 3])
                B = -torch.sum(s*unaries, dim=[1, 2, 3]) + torch.sum(x*Pr, dim=[1, 2, 3])
                batch_size, c, h, w = unaries.shape
                alphas = []
                for idx in range(batch_size):
                    # solve for each batch item: argmin_{alpha in [0, 1]} 1/2A*alpha^2 + B*alpha
                    a = A[idx]
                    b = B[idx]
                    if a <= 0:
                        alpha = 0.0 if 0.5*a + b > 0 else 1.0
                    else: # if -b/a is in [0, 1] then take it, otherwise take either 0 or 1
                        alpha = min(max(-b/a, 0.0), 1.0)
                    # print(f'{i}) a = {a}, b = {b}, alpha = {alpha}')
                    alpha = torch.ones((c, h, w), device=unaries.device)*alpha
                    alphas.append(alpha)
                # Make it (B, C, H, W)
                alpha = torch.stack(alphas)
            
            x = x + s*alpha
            # New Px: P*(x + alpha*r)
            Px = Px + Pr*alpha

        if self.print_energy or return_energy:
            e = self.energy(x, image, unaries, pairwise=Px)
            e_discrete = self.energy_discrete(x, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')
        
        if x0_weight > 0:
            # print(f'x0_weight: {x0_weight}')
            x = x_weight*x + x0_weight*x0

        if self.output_logits:
            # De-softmax to return logits
            x = torch.log(x + 1e-9)
        
        if return_energy:
            return x, energies, energies_discrete
        return x


    def frank_wolfe(self, image, unaries, return_energy=False):
        """Frank-Wolfe algorithm
        """

        if self.iterations < 1:
            return unaries

        early_stopped = False
        x0_weight = self.fw_x0_weight
        x_weight = 1.0 - x0_weight

        # initialization
        x0 = F.softmax(unaries, dim=1)
        x = x0
        Px = self.compute_pairwise(x, image)

        # set the q_values
        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)
        for i in range(self.iterations):
            
            if self.print_energy or return_energy:
                e = self.energy(x_weight*x + x0_weight*x0, image, unaries, pairwise=Px)
                e_discrete = self.energy_discrete(x_weight*x + x0_weight*x0, image, unaries)
                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # Gradient of energy
            s = Px - unaries

            if self.params.regularizer is None:
                # Minimal solution of <s, gradient>
                s = torch.argmin(s, dim=1, keepdim=True)
                s = torch.zeros(unaries.shape, device=unaries.device).scatter_(1, s, 1)
                s.requires_grad_(True)
            elif self.params.regularizer == 'negentropy':
                if self.fw_lambda != 1:
                    s = F.softmax(-s/self.fw_lambda, dim=1)
                else:
                    s = F.softmax(-s, dim=1)
            elif self.params.regularizer == 'l2':
                if self.fw_lambda != 1:
                    s = sparsemax(-s/self.fw_lambda, dim=1)
                else:
                    s = sparsemax(-s, dim=1)
            else:
                raise NotImplementedError

            # Update x = x + alpha*(s - x)
            s = s - x # r = s - x
            if self.params.scheme == 'fixed' or self.params.scheme == 'standard':
                if self.params.scheme == 'standard':
                    alpha = 2.0/(2.0 + i)
                else:
                    alpha = self.params.stepsize
                x = x + s*alpha
                if i < self.iterations - 1 or self.print_energy or return_energy:
                    Px = self.compute_pairwise(x, image)
            else:
                Pr = self.compute_pairwise(s, image)
                A = torch.sum(s*Pr, dim=[1, 2, 3])
                B = -torch.sum(s*unaries, dim=[1, 2, 3]) + torch.sum(x*Pr, dim=[1, 2, 3])
                batch_size, c, h, w = unaries.shape
                alphas = []
                for idx in range(batch_size):
                    # solve for each batch item: argmin_{alpha in [0, 1]} 1/2A*alpha^2 + B*alpha
                    a = A[idx]
                    b = B[idx]
                    if a <= 0:
                        alpha = 0.0 if 0.5*a + b > 0 else 1.0
                    else: # if -b/a is in [0, 1] then take it, otherwise take either 0 or 1
                        alpha = min(max(-b/a, 0.0), 1.0)
                    # print(f'{i}) a = {a}, b = {b}, alpha = {alpha}')
                    alpha = torch.ones((c, h, w), device=unaries.device)*alpha
                    alphas.append(alpha)
                # Make it (B, C, H, W)
                alpha = torch.stack(alphas)
                if torch.sum(alpha) <= 0:
                    # print(f'Early stopping')
                    early_stopped = True
                    for idx in range(i+1, self.iterations + 1):
                        energies[idx] = energies[i]
                        energies_discrete[idx] = energies_discrete[i]
                    break
                x = x + s*alpha
                # New Px: P*(x + alpha*r)
                if i < self.iterations - 1 or self.print_energy or return_energy:
                    Px = Px + Pr*alpha

                # Check discreteness
                # savemat(torch.max(x[0], dim=0)[0].cpu(), path=f'tests/fw_x_{i+1}.png')

        if x0_weight > 0:
            x = x_weight*x + x0_weight*x0

        if not early_stopped and (self.print_energy or return_energy):
            e = self.energy(x, image, unaries, pairwise=Px)
            e_discrete = self.energy_discrete(x, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')
        
        if self.output_logits:
            # De-softmax to return logits
            x = torch.log(x + 1e-9)
        
        if return_energy:
            return x, energies, energies_discrete
        return x


    def proximal_gradient(self, image, unaries, return_energy=False):
        """Accelerated proximal gradient algorithm (FISTA)
        """

        if self.iterations < 1:
            return unaries

        early_stopped = False

        x0_weight = self.x0_weight
        x_weight = 1.0 - x0_weight
        # initialization
        x0 = F.softmax(unaries, dim=1)
        x = x0
        y = x
        Py = self.compute_pairwise(y, image)
        t = 1
        L = 1.0
        alpha = 0

        # set the q_values
        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)
        for i in range(self.iterations):
            
            if self.print_energy or return_energy:
                if i == 0:
                    Px = Py
                else:
                    Px = (1.0/(1.0 + alpha))*(Py + alpha*Px)
                e = self.energy(x, image, unaries, pairwise=Px)
                e_discrete = self.energy_discrete(x, image, unaries)
                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # Projection
            x_prev = x
            x = sparsemax(y - (1.0/L)*(Py - unaries), dim=1)

            if i < self.iterations - 1:
                t_prev = t
                t = 0.5*(1.0 + np.sqrt(1.0 + 4.0*t_prev**2))
                alpha = (t_prev - 1)/t
                y = x + (x - x_prev)*alpha
                Py = self.compute_pairwise(y, image)


        if self.print_energy or return_energy:
            e = self.energy(x, image, unaries)
            e_discrete = self.energy_discrete(x, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')
        
        if x0_weight > 0:
            # print(f'x0_weight: {x0_weight}')
            x = x_weight*x + x0_weight*x0

        if self.output_logits:
            # De-softmax to return logits
            x = torch.log(x + 1e-9)
        
        if return_energy:
            return x, energies, energies_discrete
        return x
    

    def admm(self, image, unaries, return_energy=False):
        """
        Decomposition: 0.5x'Px + uz = 0.5x'Pz + 0.5ux + 0.5uz   
        """
        # print(f'image = {image.shape}, unaries = {unaries.shape}')

        rho = self.rho

        # Initialization
        classes = unaries.shape[1]
        y = torch.zeros_like(unaries)
        if self.params.init == 'zeros':
            x = torch.zeros_like(unaries)
        elif self.params.init == 'uniform':
            x = torch.ones_like(unaries)/classes
        elif self.params.init == 'unaries': # normalized unaries
            max_abs = torch.max(torch.abs(unaries), dim=1, keepdim=True)[0]
            x = unaries / max_abs
        elif self.params.init == 'unaries_solution': # solution of unaries, one-hot
            x = torch.argmax(unaries, dim=1, keepdim=True)
            x = torch.zeros(unaries.shape, device=unaries.device).scatter_(1, x, 1)
        elif self.params.init == 'softmax':
            x0_weight = self.x0_weight
            x_weight = 1.0 - x0_weight
            # initialization
            x0 = F.softmax(unaries, dim=1)
            x = x0
        elif self.params.init == 'mixed':
            if self.params.projection == 'bregman':
                x = torch.ones_like(unaries)/classes
                y = (0.5 - rho)*unaries + self.compute_pairwise(rho*F.softmax(unaries, dim=1) - 0.5*x, image)
            else:
                z = torch.ones_like(unaries)/classes
                x = z*(self.rho/2)
                y = unaries*(0.5 - self.rho) + z*self.rho
                # x = z
                # y = z*0.5
        else:
            raise NotImplementedError

        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)

        for i in range(self.iterations):
            # 1. UPDATE X

            x_prev = x

            # 1.a. Compute P*z
            x = self.compute_pairwise(x_prev, image)
            
            if self.print_energy or return_energy:
                if i == 0 and self.params.init != 'softmax':
                    e = self.energy(F.softmax(unaries, dim=1), image, unaries)
                    e_discrete = self.energy_discrete(F.softmax(unaries, dim=1), image, unaries)
                else:    
                    e = self.energy(x_prev, image, unaries, pairwise=x)
                    e_discrete = self.energy_discrete(x_prev, image, unaries)

                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # 1.b. Compute a = (-0.5*u - 0.5*Pz - y)/rho
            x = (0.5*unaries - 0.5*x - ((-1)**i)*y)/rho
            # 1.c. Compute x
            
            if self.projection == 'bregman':
                x = x_prev * torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
                x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-9)
            elif self.projection == 'euclidean':
                x = sparsemax(x_prev + x, 1)
            else:
                raise Exception(f"Projection type {self.projection} not known!")
            
            if i % 2 == 1:
                y = y + rho*(x_prev - x)

        if self.print_energy or return_energy:
            e = self.energy(x, image, unaries)
            e_discrete = self.energy_discrete(x, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')

        if x0_weight > 0:
            # print(f'x0_weight: {x0_weight}')
            x = x_weight*x + x0_weight*x0

        if self.output_logits:
            # De-softmax to return logits
            x = torch.log(x + 1e-9)
        
        if return_energy:
            return x, energies, energies_discrete
        return x

    
    def admm_xz(self, image, unaries, return_energy=False):
        """
        Decomposition: 0.5x'Px + uz = 0.5x'Pz + 0.5ux + 0.5uz   
        """
        # print(f'image = {image.shape}, unaries = {unaries.shape}')

        rho = self.rho

        # Initialization
        classes = unaries.shape[1]
        y = torch.zeros_like(unaries)
        if self.params.init == 'zeros':
            x = torch.zeros_like(unaries)
            z = torch.zeros_like(unaries)
        elif self.params.init == 'uniform':
            x = torch.ones_like(unaries)/classes
            z = torch.ones_like(unaries)/classes
        elif self.params.init == 'unaries': # normalized unaries
            max_abs = torch.max(torch.abs(unaries), dim=1, keepdim=True)[0]
            # print(f'max_abs shape = {max_abs.shape}')
            x = unaries / max_abs
            z = unaries / max_abs
        elif self.params.init == 'unaries_solution': # solution of unaries
            one_hot = torch.argmax(unaries, dim=1, keepdim=True)
            one_hot = torch.zeros(unaries.shape, device=unaries.device).scatter_(1, one_hot, 1)
            x = one_hot
            z = one_hot
        elif self.params.init == 'softmax':
            x = F.softmax(unaries, dim=1)
            z = F.softmax(unaries, dim=1)
        elif self.params.init == 'mixed':
            if self.params.projection == 'bregman':
                x = F.softmax(unaries, dim=1)
                z = torch.ones_like(unaries)/classes
                y = (0.5 - rho)*unaries + self.compute_pairwise(rho*x - 0.5*z, image)
            else:
                z = torch.ones_like(unaries)/classes
                x = z*(self.rho/2)
                y = -unaries*(self.rho - 0.5) + z*self.rho
                # x = z
                # y = z*0.5
        else:
            raise NotImplementedError

        # if self.rho is None:
        #     # conv = self.rho_conv(F.softmax(unaries, dim=1))
        #     # print(f'conv max = {torch.max(conv)}, conv min = {torch.min(conv)}')
        #     # rho = 1.0 + conv
        #     rho = 1.0 + torch.max(unaries, dim=1, keepdim=True)[0]
        #     # print(f'rho.shape = {rho.shape}')
        # else:
        #     rho = self.rho


        # if self.admm_rho_type == 'fixed':
        #     rho = self.rho
        # elif self.admm_rho_type == 'scalar':
        #     rho = self.rho

        #     # # print(f'rho.shape = {rho.shape}')
        #     # M = torch.max(x, dim=1)[0]
        #     # # scale = torch.ones_like(M)
        #     # # scale.masked_fill(M > 0.8, 2.0)
        #     # # print(f'rho.shape = {rho.shape}')

        #     # a = (self.rho_scale - 1.0)/(1.0 - 1.0/self.classes)
        #     # b = self.rho_scale - a
        #     # scale = a*M + b
        #     # rho = (scale*rho).unsqueeze(1)
        # elif self.admm_rho_type == 'vector':
        #     rho = self.rho.repeat(unaries.shape[-2], unaries.shape[-1], 1).permute(2, 0, 1)
        # elif self.admm_rho_type == 'tensor':
        #     pass
        

        # print(f'rho max = {torch.max(rho)}, rho min = {torch.min(rho)}')

        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)

        for i in range(self.iterations):
            # 1. UPDATE X

            # 1.a. Compute P*z
            x = self.compute_pairwise(z, image)
            
            if self.print_energy or return_energy:
                if i == 0 and self.params.init != 'softmax':
                    e = self.energy(F.softmax(unaries, dim=1), image, unaries)
                    e_discrete = self.energy_discrete(F.softmax(unaries, dim=1), image, unaries)
                else:    
                    e = self.energy(z, image, unaries, pairwise=x)
                    e_discrete = self.energy_discrete(z, image, unaries)

                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # 1.b. Compute a = -(0.5*u + 0.5*Pz + y)/rho
            x = (0.5*unaries - 0.5*x - y)/rho
            # 1.c. Compute x
            if self.loose:
                if self.projection == 'bregman':
                    x = z*torch.exp(x - 1)
                elif self.projection == 'euclidean':
                    x = F.relu(z + x)
                else:
                    raise Exception(f"Projection type {self.projection} not known!")
            else:
                if self.projection == 'bregman':
                    x = z * torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
                    x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-9)
                elif self.projection == 'euclidean':
                    x = sparsemax(z + x, 1)
                else:
                    raise Exception(f"Projection type {self.projection} not known!")

            # if self.projection == 'bregman':
            #     x = z * torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
            #     x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-9)
            # elif self.projection == 'euclidean':
            #     x = sparsemax(z + x, 1)
            # else:
            #     raise Exception(f"Projection type {self.projection} not known!")

            # 2. UPDATE Z

            # 2.a. Compute P*x
            # z = (self.spatial_compat(self.spatial_weight(self.spatial_filter(x, image))) + 
            #      self.bilateral_compat(self.bilateral_weight(self.bilateral_filter(x, image))))
            # z *= self.pairwise_scale
            z = self.compute_pairwise(x, image)
            
            # 2.b. Compute b = -(0.5*u + 0.5*Px - y)/rho
            z = (0.5*unaries - 0.5*z + y)/rho
            # 2.c. Compute z
            if self.projection == 'bregman':
                z = x * torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])
                z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-9)
            elif self.projection == 'euclidean':
                z = sparsemax(x + z, 1)
            else:
                raise Exception(f"Projection type {self.projection} not known!")
            
            # if self.loose:
            #     if self.projection == 'bregman':
            #         z = x*torch.exp(z - 1)
            #     elif self.projection == 'euclidean':
            #         z = F.relu(x + z)
            #     else:
            #         raise Exception(f"Projection type {self.projection} not known!")
            # else:
            #     if self.projection == 'bregman':
            #         z = x * torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])
            #         z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-9)
            #     elif self.projection == 'euclidean':
            #         z = sparsemax(x + z, 1)
            #     else:
            #         raise Exception(f"Projection type {self.projection} not known!")

            # 3 UPDATE Y
            y = y + rho*(x - z)


            # Check discreteness
            # savemat(torch.max(x[0], dim=0)[0].cpu(), path=f'tests/admm_x_{i+1}.png')
            # savemat(torch.max(z[0], dim=0)[0].cpu(), path=f'tests/admm_z_{i+1}.png')

        if self.print_energy or return_energy:
            e = self.energy(z, image, unaries)
            e_discrete = self.energy_discrete(z, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')

        if self.output_logits:
            # De-softmax to return logits
            z = torch.log(z + 1e-9)
        
        if return_energy:
            return z, energies, energies_discrete
        return z


    def admm2(self, image, unaries, return_energy=False):
        # print(f'image = {image.shape}, unaries = {unaries.shape}')

        # Initialization
        classes = unaries.shape[1]
        y = torch.zeros_like(unaries)
        if self.params.init == 'zeros':
            x = torch.zeros_like(unaries)
            z = torch.zeros_like(unaries)
        elif self.params.init == 'uniform':
            x = torch.ones_like(unaries)/classes
            z = torch.ones_like(unaries)/classes
        elif self.params.init == 'unaries': # normalized unaries
            max_abs = torch.max(torch.abs(unaries), dim=1, keepdim=True)[0]
            # print(f'max_abs shape = {max_abs.shape}')
            x = unaries / max_abs
            z = unaries / max_abs
        elif self.params.init == 'unaries_solution': # solution of unaries
            one_hot = torch.argmax(unaries, dim=1, keepdim=True)
            one_hot = torch.zeros(unaries.shape, device=unaries.device).scatter_(1, one_hot, 1)
            x = one_hot
            z = one_hot
        elif self.params.init == 'softmax':
            x = F.softmax(unaries, dim=1)
            z = F.softmax(unaries, dim=1)
        elif self.params.init == 'mixed':
            if self.params.projection == 'bregman':
                x = F.softmax(unaries, dim=1)
                z = torch.ones_like(unaries)/classes
                y = self.compute_pairwise(x - z, image)
            else:
                z = torch.ones_like(unaries)/classes
                x = z*(self.rho/2)
                y = -unaries*(self.rho - 0.5) + z*self.rho
                # x = z
                # y = z*0.5
        else:
            raise NotImplementedError

        # max_abs = torch.max(torch.abs(unaries), dim=1, keepdim=True)[0]
        # # print(f'max_abs shape = {max_abs.shape}')
        # x = unaries / max_abs
        # z = unaries / max_abs

        # if self.rho is None:
        #     # conv = self.rho_conv(F.softmax(unaries, dim=1))
        #     # print(f'conv max = {torch.max(conv)}, conv min = {torch.min(conv)}')
        #     # rho = 1.0 + conv
        #     rho = 1.0 + torch.max(unaries, dim=1, keepdim=True)[0]
        #     # print(f'rho.shape = {rho.shape}')
        # else:
        #     rho = self.rho

        rho = self.rho

        # if self.admm_rho_type == 'fixed':
        #     rho = self.rho
        # elif self.admm_rho_type == 'scalar':
        #     rho = self.rho

        #     # # print(f'rho.shape = {rho.shape}')
        #     # M = torch.max(x, dim=1)[0]
        #     # # scale = torch.ones_like(M)
        #     # # scale.masked_fill(M > 0.8, 2.0)
        #     # # print(f'rho.shape = {rho.shape}')

        #     # a = (self.rho_scale - 1.0)/(1.0 - 1.0/self.classes)
        #     # b = self.rho_scale - a
        #     # scale = a*M + b
        #     # rho = (scale*rho).unsqueeze(1)
        # elif self.admm_rho_type == 'vector':
        #     rho = self.rho.repeat(unaries.shape[-2], unaries.shape[-1], 1).permute(2, 0, 1)
        # elif self.admm_rho_type == 'tensor':
        #     pass
        

        # print(f'rho max = {torch.max(rho)}, rho min = {torch.min(rho)}')

        if self.ergodic:
            s = z

        energies = torch.zeros(self.iterations + 1)
        energies_discrete = torch.zeros(self.iterations +1)

        for i in range(self.iterations):

            # rho = self.rho*1.5**i

            # 1. UPDATE X

            # 1.a. Compute P*z
            # x = self.spatial_compat(self.spatial_weight(self.spatial_filter(z, image)))
            # x = x + self.bilateral_compat(self.bilateral_weight(self.bilateral_filter(z, image)))
            # x = x * self.pairwise_scale
            x = self.compute_pairwise(z, image)
            
            if self.print_energy or return_energy:
                if not self.ergodic:
                    if i == 0 and self.params.init != 'softmax':
                        e = self.energy(F.softmax(unaries, dim=1), image, unaries)
                        e_discrete = self.energy_discrete(F.softmax(unaries, dim=1), image, unaries)
                    else:    
                        e = self.energy(z, image, unaries, pairwise=x)
                        e_discrete = self.energy_discrete(z, image, unaries)
                else:
                    e = self.energy(s/(i+1), image, unaries)
                    e_discrete = self.energy_discrete(s/(i+1), image, unaries)
                energies[i] = e
                energies_discrete[i] = e_discrete
                if self.print_energy:
                    print(f'{i}) e = {e}, e_discrete = {e_discrete}')

            # 1.b. Compute a = -(0.5*u + 0.5*Pz + y)/rho
            x = (0.5*unaries - 0.5*x - y)/rho
            # 1.c. Compute x
            # if self.loose:
            #     if self.projection == 'bregman':
            #         x = z*torch.exp(x - 1)
            #     elif self.projection == 'euclidean':
            #         x = F.relu(z + x)
            #     else:
            #         raise Exception(f"Projection type {self.projection} not known!")
            # else:
            #     if self.projection == 'bregman':
            #         x = z * torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
            #         x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-9)
            #     elif self.projection == 'euclidean':
            #         x = sparsemax(z + x, 1)
            #     else:
            #         raise Exception(f"Projection type {self.projection} not known!")

            if self.projection == 'bregman':
                x = z * torch.exp(x - torch.max(x, dim=1, keepdim=True)[0])
                x = x / (torch.sum(x, dim=1, keepdim=True) + 1e-9)
            elif self.projection == 'euclidean':
                x = sparsemax(z + x, 1)
            else:
                raise Exception(f"Projection type {self.projection} not known!")

            # 2. UPDATE Z

            # 2.a. Compute P*x
            # z = (self.spatial_compat(self.spatial_weight(self.spatial_filter(x, image))) + 
            #      self.bilateral_compat(self.bilateral_weight(self.bilateral_filter(x, image))))
            # z *= self.pairwise_scale
            z = self.compute_pairwise(x, image)
            
            # 2.b. Compute b = -(0.5*u + 0.5*Px - y)/rho
            z = (0.5*unaries + y - 0.5*z)/rho
            # 2.c. Compute z
            if self.projection == 'bregman':
                z = x * torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])
                z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-9)
            elif self.projection == 'euclidean':
                z = sparsemax(x + z, 1)
            else:
                raise Exception(f"Projection type {self.projection} not known!")
            
            if self.loose:
                if self.projection == 'bregman':
                    z = x*torch.exp(z - 1)
                elif self.projection == 'euclidean':
                    z = F.relu(x + z)
                else:
                    raise Exception(f"Projection type {self.projection} not known!")
            else:
                if self.projection == 'bregman':
                    z = x * torch.exp(z - torch.max(z, dim=1, keepdim=True)[0])
                    z = z / (torch.sum(z, dim=1, keepdim=True) + 1e-9)
                elif self.projection == 'euclidean':
                    z = sparsemax(x + z, 1)
                else:
                    raise Exception(f"Projection type {self.projection} not known!")

            # 3 UPDATE Y
            y = y + rho*(x - z)

            if self.ergodic:
                s = s + z

            # Check discreteness
            # savemat(torch.max(x[0], dim=0)[0].cpu(), path=f'tests/admm_x_{i+1}.png')
            # savemat(torch.max(z[0], dim=0)[0].cpu(), path=f'tests/admm_z_{i+1}.png')

        if self.ergodic:
            z = s/(1+self.iterations)

        if self.print_energy or return_energy:
            e = self.energy(z, image, unaries)
            e_discrete = self.energy_discrete(z, image, unaries)
            energies[i+1] = e
            energies_discrete[i+1] = e_discrete
            if self.print_energy:
                print(f'{i+1}) e = {e}, e_discrete = {e_discrete}')

        if self.output_logits:
            # De-softmax to return logits
            z = torch.log(z + 1e-9)
        
        if return_energy:
            return z, energies, energies_discrete
        return z



class GaussianCRF(GeneralCRF):
    """Base class for Gaussian CRFs

    Args:
        connectivity (string, optional):
            - 'dense': fully-connected
            - 'truncated':
            - 'grid'
        init (string, optional):
            - 'potts': Potts model
            - 'potts-random':
            - 'random'
    """
    def __init__(self, classes, alpha, beta, gamma,
                spatial_weight=-1, bilateral_weight=-1, compatibility=-1,
                init='potts', bias=False,
                solver='mf', iterations=5, params=None,
                ergodic=False, x0_weight=0.0, output_logits=True, print_energy=False,
                **kargs):

        super().__init__(solver, iterations, params,
                ergodic, x0_weight, output_logits, print_energy)
        
        self.classes = classes
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spatial_filter = None
        self.bilateral_filter = None

        # spatial_weight = 1.0 if spatial_weight < 0 else spatial_weight
        # bilateral_weight = 1.0 if bilateral_weight < 0 else bilateral_weight
        
        # Learnable weights
        self.spatial_weight = nn.Conv2d(classes, classes, (1, 1), bias=bias)
        self.bilateral_weight = nn.Conv2d(classes, classes, (1, 1), bias=bias)
        self.compatibility = nn.Conv2d(classes, classes, (1, 1), bias=bias)

        # if potts and spatial_weight < 0:
        #     print(f'Potts: reset spatial_weight to 1.0.')
        #     spatial_weight = 1.0
        # if spatial_weight >= 0:
        #     self.spatial_weight.weight.data.fill_(spatial_weight)
        #     if self.spatial_weight.bias is not None:
        #         self.spatial_weight.bias.data.zero_()

        if init in ['potts', 'potts-random']:
            spatial_weight = 1.0 if spatial_weight < 0 else spatial_weight
            bilateral_weight = 1.0 if bilateral_weight < 0 else bilateral_weight
            compatibility = 1.0 if compatibility < 0 else compatibility
        else:
            print('Random initialization. Kernel weights and compat weights are ignored.')
            spatial_weight = -1
            bilateral_weight = -1
            compatibility = -1


        if init == 'potts':
            # Remove random weights
            print(f'Potts: remove random weights.')
            self.spatial_weight.weight.data.zero_()
            if self.spatial_weight.bias is not None:
                self.spatial_weight.bias.data.zero_()
            self.bilateral_weight.weight.data.zero_()
            if self.bilateral_weight.bias is not None:
                self.bilateral_weight.bias.data.zero_()
            self.compatibility.weight.data.zero_()
            if self.compatibility.bias is not None:
                self.compatibility.bias.data.zero_()

        if spatial_weight >= 0:
            # print(f'Initialize spatial_weight to {spatial_weight}')
            # self.spatial_weight.weight.data.fill_(spatial_weight)
            # self.spatial_weight.bias.data.zero_()
            # print(f'Initialize spatial_weight to Potts {spatial_weight}')
            # self.spatial_weight.weight.data.fill_(0.0)[...,0,0].fill_diagonal_(-spatial_weight)
            print(f'Add {spatial_weight} to spatial_weight diagonal')
            # self.spatial_weight.weight.data[...,0,0].fill_diagonal_(-spatial_weight)
            self.spatial_weight.weight.data[...,0,0].diagonal().add_(spatial_weight)
        
        if bilateral_weight >= 0:
            # print(f'Initialize bilateral_weight to {bilateral_weight}')
            # self.bilateral_weight.weight.data.fill_(bilateral_weight)
            # self.bilateral_weight.bias.data.zero_()
            
            # print(f'Initialize bilateral_weight to Potts {bilateral_weight}')
            # self.bilateral_weight.weight.data.fill_(0.0)[...,0,0].fill_diagonal_(-bilateral_weight)
            
            print(f'Add {bilateral_weight} to bilateral_weight diagonal')
            # self.bilateral_weight.weight.data[...,0,0].fill_diagonal_(-bilateral_weight)
            self.bilateral_weight.weight.data[...,0,0].diagonal().add_(bilateral_weight)

        if compatibility >= 0:
            print(f'Add {-compatibility} to compatibility diagonal')
            # self.compat.weight.data.fill_(compat)[...,0,0].fill_diagonal_(0)
            self.compatibility.weight.data[...,0,0].diagonal().add_(-compatibility)

        # # # One can use a matrix instead of convolution. This seems faster.
        # self.spatial_weight = torch.nn.Parameter(torch.eye(classes)*spatial_weight)
        # self.bilateral_weight = torch.nn.Parameter(torch.eye(classes)*bilateral_weight)
        # # potts = torch.ones(classes, classes).fill_diagonal_(0)
        # # self.spatial_weight = torch.nn.Parameter(potts)
        # compatibility = 1.0 if compatibility < 0 else compatibility
        # self.compatibility = torch.nn.Parameter(-torch.eye(classes)*compatibility)


    def compute_pairwise(self, x, image):
        """Compute P * x
        """
        Px = self.spatial_weight(self.spatial_filter(x, image))
        # print(f'spatial_out norm = {torch.norm(self.spatial_filter(x, image))}')
        Px = Px + self.bilateral_weight(self.bilateral_filter(x, image))
        # print(f'bilateral_out norm = {torch.norm(self.bilateral_filter(x, image))}')
        # Px = self.spatial_weight(self.spatial_filter(x, image)) + self.bilateral_weight(self.bilateral_filter(x, image))
        # print(f'message_passing norm = {torch.norm(Px)}')
        Px = self.compatibility(Px)
        # print(f'pairwise norm = {torch.norm(Px)}')
        return Px


    def compute_pairwise_mat(self, x, image):
        """Compute P * x
        """
        N, C, H, W = x.shape
        Px = self.spatial_filter(x, image)
        Px = torch.matmul(self.spatial_weight, Px.reshape(N, C, -1))
        x = self.bilateral_filter(x, image)
        # print(f'x shape = {x.shape}, bilateral shape = {self.bilateral_weight.shape}')
        x = torch.matmul(self.bilateral_weight, x.reshape(N, C, -1))
        Px = Px + x
        # print(f'x out shape = {x.shape}, Px shape = {Px.shape}')
        Px = torch.matmul(self.compatibility, Px)
        Px = Px.reshape(N, C, H, W)
        return Px


    def compute_pairwise_debug(self, x, image):
        """Compute P * x
        """
        N, C, H, W = x.shape
        # print(f'x {x.shape}')
        spatial_out = self.spatial_filter(x, image)
        # print(f'x {x.shape}: \n {x[0, :10].reshape(10, -1)[:10,:10].cpu().numpy()}')
        # print(f'spatial_out = {spatial_out.shape} \n {spatial_out[0].view(self.classes, -1)}')
        # print(f'spatial_out {spatial_out.shape}: \n {spatial_out[:10].reshape(10, -1)}')
        # print(f'spatial_out permuted {spatial_out.shape}: \n {spatial_out.permute(0,1,3,2)[0, :10].reshape(10, -1)[:10,:10].cpu().numpy()}')
        # print(f'spatial_out {spatial_out.shape}: \n {spatial_out[0, :10].reshape(10, -1)[:10,:10].cpu().numpy()}')
        # spatial_out = self.spatial_weight(spatial_out)
        # print(f'spatial_filter = {Px.shape}')
        spatial_out = torch.matmul(self.spatial_weight, spatial_out.reshape(N, C, -1)) 
        # spatial_out = spatial_out.reshape(N, C, H, W)
        # print(f'spatial_out = {spatial_out.shape} \n {spatial_out[0].view(self.classes, -1)}')
        # print(f'spatial_out norm = {torch.norm(self.spatial_filter(x, image))}')
        bilateral_out = self.bilateral_filter(x, image)
        # print(f'bilateral_out permuted {bilateral_out.shape}: \n {bilateral_out.permute(0,1,3,2)[0, :10].reshape(10, -1)[:10,:10].cpu().numpy()}')
        # print(f'bilateral_out {bilateral_out.shape}: \n {bilateral_out[0, :10].reshape(10, -1)[:10,:10].cpu().numpy()}')
        bilateral_out = torch.matmul(self.bilateral_weight, bilateral_out.reshape(N, C, -1)) 
        # bilateral_out = bilateral_out.reshape(N, C, H, W)
        # print(f'bilateral_out = {bilateral_out.shape} \n {bilateral_out[0].view(self.classes, -1)}')
        # bilateral_out = self.bilateral_weight(bilateral_out)
        # print(f'bilateral_out = {bilateral_out.shape} \n {bilateral_out[0].view(self.classes, -1)}')
        # print(f'bilateral_out norm = {torch.norm(self.bilateral_filter(x, image))}')
        # Px = self.spatial_weight(self.spatial_filter(x, image)) + self.bilateral_weight(self.bilateral_filter(x, image))
        # print(f'message_passing norm = {torch.norm(Px)}')
        # Px = self.compat(spatial_out + bilateral_out)
        # print(f'pairwise norm = {torch.norm(Px)}')
        # Px = torch.matmul(self.compat, spatial_out + bilateral_out) 
        # print(f'pairwise1 = {spatial_out[0]}')
        # print(f'pairwise2 = {bilateral_out[0]}')
        Px = spatial_out + bilateral_out
        Px = Px.reshape(N, C, H, W)
        # print(f'pairwise = {Px.shape} \n {Px[0, :2, 110:115, 55:60]}')
        return Px


class DenseGaussianCRF(GaussianCRF):
    r"""This is the fully connected CRF with Gaussian potentials proposed by
    Krähenbühl and Koltun (NeurIPS 2011).

    Args:
        classes (int): number of classes.
        alpha, beta, gamma (float, optional): standard deviations
            of the Gaussian kernels (see paper for details).
        output_logits (bool, optional): return logits instead of probabilities.
            Default: `True`.
        inference (str, optional): optimization method.
            `admm` for ADMM, `mf` for Mean Field, and `fw` for Frank-Wolfe.
            Default: `admm`.
        iterations (int, optional): number of optimization iterations. Default: 5.
        ergodic (bool, optional): return the ergodic solution: (x_1 + ... + x_N)/N.
            Default: False.
        admm_projection: 'bregman' or 'euclidean'
        admm_loose_decomposition: True means (x: non-negative, z: simplex)
            False means (x: simplex, z: simplex)
        admm_rho (float, optional): initial value for ADMM penalty. Default: 1.0.
        admm_rho_type (str, optional): `fixed`, or `learnable-scalar`, or
            `learnable-tensor`
    
    Default: alpha=80.0, beta=13.0, gamma=3.0,
                spatial_compat=3.0, bilateral_compat=10.0,
                spatial_weight=1.0, bilateral_weight=1.0
    
    Examples::
        >>> input = torch.randn(20, 16, 50, 32, 16)
        >>> # pool of cubic window of size=3, and target output size 13x12x11
        >>> F.fractional_max_pool3d(input, 3, output_size=(13, 12, 11))
        >>> # pool of cubic window and target output size being half of input size
        >>> F.fractional_max_pool3d(input, 3, output_ratio=(0.5, 0.5, 0.5))

    .. _`Efficient inference in fully connected crfs with gaussian edge potentials`:
        https://arxiv.org/abs/1210.5644
        """
    def __init__(self, **kargs):
        super().__init__(**kargs)
        # Spatial kernel weights
        self.spatial_filter = PermutohedralLayer(bilateral=False,
                                                theta_alpha=self.alpha,
                                                theta_beta=self.beta,
                                                theta_gamma=self.gamma)
        # Bilateral kernels
        self.bilateral_filter = PermutohedralLayer(bilateral=True,
                                                theta_alpha=self.alpha,
                                                theta_beta=self.beta,
                                                theta_gamma=self.gamma)


class TruncatedGaussianCRF(GaussianCRF):
    def __init__(self, classes, alpha, beta, gamma, window=-1, blur=1, **kargs):
        super().__init__(classes, alpha, beta, gamma, **kargs)
        self.window = window
        self.blur = blur

    def forward(self, image, unaries):
        """The bilateral filters are created on the fly
        """
        bsz, classes, height, width = unaries.shape

        # Shape: (N, 2, H, W), the 2 dimensions correspond to x and y coordinates
        spatial = create_position_feats((height, width), bs=bsz)
        spatial = spatial.to(unaries.device)
        # print(f'spatial_feats {spatial_feats.shape} \n {spatial_feats}')

        # Shape: (N, 1, filter_size, filter_size, H, W)
        spatial = create_conv_filters(spatial / self.gamma,
                                filter_size=self.window, blur=self.blur)

        self.spatial_filter = lambda x, im: perform_filtering(x, spatial, blur=self.blur)

        # if True:
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225]
            # mean = torch.as_tensor(mean, dtype=image.dtype, device=image.device)
            # std = torch.as_tensor(std, dtype=image.dtype, device=image.device)
            # image = image.permute(0, 2, 3, 1)
            # image.mul_(std).add_(mean)
            # image = image*255
            # image = image.permute(0, 3, 1, 2)

        # print(f'image shape = {image.shape}')
        # print(f'spatial_feats shape = {spatial_feats.shape}')

        # Bilateral filter
        # Shape: (N, 5, H, W), the 5 dimensions correspond (x, y, r, g, b)
        bilateral = create_position_feats((height, width), bs=bsz)
        bilateral = bilateral.to(unaries.device)
        # print(f'bilateral dtype = {bilateral.dtype}, image dtype = {image.dtype}')
        bilateral = torch.cat([bilateral / self.alpha, image / self.beta], dim=1)
        # Shape: (N, 1, filter_size, filter_size, H, W)
        bilateral = create_conv_filters(bilateral,
                                filter_size=self.window, blur=self.blur)

        self.bilateral_filter = lambda x, im: perform_filtering(x, bilateral, blur=self.blur)

        return super().forward(image, unaries)


class DenseGaussianCRF_cpu(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, classes, height, width,
                alpha=80.0, beta=13.0, gamma=3.0,
                spatial_weight=-1, bilateral_weight=-1, compat=-1,
                output_logits=True, print_energy=False,
                inference='admm', iterations=5, ergodic=False,
                admm_projection='bregman', admm_loose=False, admm_rho=1.0,
                admm_rho_type='fixed'):
        """
        https://github.com/lucasb-eyer/pydensecrf
        The compat argument can be any of the following:
            scalar: PottsCompatibility.
            1D array: DiagonalCompatibility.
            2D array: MatrixCompatibility.

        Possible values for the kernel argument are:
            CONST_KERNEL
            DIAG_KERNEL (the default)
            FULL_KERNEL
            This specifies the kernel's precision-matrix Λ(m)
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.spatial_weight = 1.0 if spatial_weight < 0 else spatial_weight
        self.bilateral_weight = 1.0 if bilateral_weight < 0 else bilateral_weight
        self.compat = 1.0 if compat < 0 else compat


    @torch.no_grad()
    def forward(self, image, unaries, iterations=5, output_logits=True):
        """Perform CRF inference on the CPU using (non-learnable) pydensecrf
        unaries: logits from CNN

        Shape:
            image: (N, C, H, W)
            unaries: (N, L, H, W)
        """
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        print(f'unaries = {unaries.shape}')
        # print(unaries[0,:, 105:115, 50:60])
        bsz, classes, height, width = unaries.shape

        device = unaries.device
        
        # pyDenseCRF
        d = dcrf.DenseCRF2D(width, height, classes)  # width, height, nlabels

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(gamma, gamma),
            compat=self.compat*self.spatial_weight, kernel=dcrf.DIAG_KERNEL,
            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # Unaries: (batch, labels, height, width)
        unaries = np.ascontiguousarray(unaries.cpu().numpy())
        # Convert (N, C, H, W) to (N, W, H, C)
        # image = unnormalize(image, mean=[0.485, 0.456, 0.406],
        #                         std=[0.229, 0.224, 0.225])
        # image = image*255
        # image = np.swapaxes(image.cpu().numpy(), 1, 3)
        # image = np.ascontiguousarray(image, dtype=np.uint8)
        # print(f'image = {image.shape}')

        Q = np.zeros((bsz, classes, height, width))

        # For each item in batch
        for idx in range(bsz):
            U = -unaries[idx].reshape((classes,-1)) # Needs to be flat.
            d.setUnaryEnergy(U)

            # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
            # im is an image-array, e.g. im.dtype == np.uint8
            # and im.shape == (width, height, channels)
            # print(f'before unnormalized {image[idx].shape} = {image[idx][:, 110:115, 55:60]}')
            # tensor = unnormalize(image[idx], mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
            # tensor = tensor*255
            # # print(f'after unnormalized {tensor.shape} = {tensor[:, 110:115, 55:60]}')
            # # tensor = np.swapaxes(tensor.cpu().numpy(), 0, 2)
            # # print(f'after swap axes {tensor.shape} = {tensor[110:115, 55:60, :]}')
            # tensor = np.transpose(tensor.cpu().numpy(), (1, 2, 0))
            # tensor = np.ascontiguousarray(tensor, dtype=np.uint8)


            tensor = np.swapaxes(image[idx].cpu().numpy(), 0, 2)
            # print(f'after transpose {tensor.shape} = {tensor[110:115, 55:60, :]}')
            # (C, H, W) to (W, H, C)
            d.addPairwiseBilateral(sxy=(alpha, alpha),
                                srgb=(beta, beta, beta),
                                rgbim=tensor,
                                compat=self.compat*self.bilateral_weight,
                                kernel=dcrf.DIAG_KERNEL,
                                normalization=dcrf.NORMALIZE_SYMMETRIC)

            # q = np.array(d.inference(iterations)).reshape((classes, height, width))

            q, tmp1, tmp2 = d.startInference()
            for i in range(iterations):
                print("KL-divergence at {}: {}".format(i, d.klDivergence(q)))
                d.stepInference(q, tmp1, tmp2)
            Q[idx] = np.array(q).reshape((classes, height, width))

        # Convert to Tensor for compatibility with other layers
        Q = torch.from_numpy(Q).to(device)

        if not output_logits:
            Q = F.softmax(Q, dim=1)

        return Q
