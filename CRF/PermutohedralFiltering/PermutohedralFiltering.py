# See http://graphics.stanford.edu/papers/permutohedral/ for details
# https://github.com/idofr/pymutohedral_lattice for understanding

import torch
import Permutohedral
import Permutohedral_gpu


class PermutohedralFiltering(torch.autograd.Function):

    """
    This function is used for the permutohedral filtering. This methode consists of four steps:
    1. Generating Position vektors (I dont know what to do )
    2. Splatting: Define the enclosing simplex and compute barycentric weights. (complex and I dont know what to do)
    3. Blurring: Gaussian Blur with [1 2 1] along each lattice direction.
    4. Inverse of Splatting
    """
    @staticmethod
    def forward(ctx, x, image, bilateral, 
                theta_alpha, theta_beta, theta_gamma):
        assert len(x.shape) == 4

        ctx.save_for_backward(image)
        ctx.bilateral = bilateral
        ctx.theta_alpha = theta_alpha
        ctx.theta_beta = theta_beta
        ctx.theta_gamma = theta_gamma

        if torch.cuda.is_available():
            output = Permutohedral_gpu.forward(x, image, bilateral,
                              theta_alpha, theta_beta, theta_gamma)
            
        else:
            output = Permutohedral.forward(x, image, bilateral,
                              theta_alpha, theta_beta, theta_gamma)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        image, = ctx.saved_tensors
        if torch.cuda.is_available():
            grad_x = Permutohedral_gpu.backward(grad_output, image,
               ctx.bilateral, ctx.theta_alpha, ctx.theta_beta, ctx.theta_gamma)
        else:
            grad_x = Permutohedral.backward(grad_output, image,
               ctx.bilateral, ctx.theta_alpha, ctx.theta_beta, ctx.theta_gamma)

        return grad_x, torch.zeros_like(image), None, None, None, None


class PermutohedralLayer(torch.nn.Module):
    def __init__(self, bilateral, theta_alpha, theta_beta, theta_gamma, nhwc=True):
        super(PermutohedralLayer, self).__init__()
        self.bilateral = bilateral
        self.theta_alpha = theta_alpha
        self.theta_beta = theta_beta
        self.theta_gamma = theta_gamma
        self.nhwc = nhwc

    def forward(self, x, image):
        permuted = PermutohedralFiltering.apply(x.permute(0,2,3,1).contiguous(), image.permute(0,2,3,1).contiguous(), self.bilateral,
            self.theta_alpha, self.theta_beta, self.theta_gamma)
        
        return permuted.permute(0,3,1,2)

    def forward_old(self, x, image):
        if self.nhwc:
            x = x.permute(0,2,3,1).contiguous()
            image = image.permute(0,2,3,1).contiguous()

        permuted = PermutohedralFiltering.apply(x, image, self.bilateral,
            self.theta_alpha, self.theta_beta, self.theta_gamma)
        
        if self.nhwc:
            permuted = permuted.permute(0,3,1,2)

        return permuted