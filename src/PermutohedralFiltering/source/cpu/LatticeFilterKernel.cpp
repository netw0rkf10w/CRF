/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis, Sebastian Hahn

Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all
        copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
        AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/
// changed such that it works with pytorch

#include "PermutohedralLatticeCPU.h"
#include "LatticeFilter.hpp"

#include <torch/extension.h>


at::Tensor
LatticeFilter_calculate(
    const at::Tensor& input_tensor,
    const at::Tensor& image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0,
    bool backward=false
)
{
    // calculate dimensions; dimension 0 is batch; dim 1 is channel
    int rank = input_tensor.ndimension();
    int n_spatial_dims = rank - 2;
    int pd;

    auto batch_size = static_cast<int>(input_tensor.size(0));
    auto n_input_channels = static_cast<int>(input_tensor.size(1));
    auto spatial_dims = new int[n_spatial_dims];

    int num_super_pixels{1};
    for (int i = 0; i < n_spatial_dims; i++){
        auto dim_size = static_cast<int>(input_tensor.size(i + 2)); // ignore the first two channels (batch and color)
        num_super_pixels *= dim_size;
        spatial_dims[i] = dim_size;
    }

    int vd = n_input_channels + 1;
    float spatial_std;
    float features_std;
    int n_reference_channels;

    if(bilateral){
        assert(reference_image_tensor.dims() == rank);
        n_reference_channels = static_cast<int>(image_tensor.size(1));
        pd = n_reference_channels + n_spatial_dims;
        spatial_std = theta_alpha;
        features_std = theta_beta;
    }
    else
    {
        pd = n_spatial_dims;
        n_reference_channels = 0; //set to zero so ComputeKernel does not use reference image channels
        spatial_std = theta_gamma;
        features_std = -1; //does not matter
    }

        // Allocate kernel positions and calculate them

        at::Tensor positions = at::zeros({batch_size * num_super_pixels * pd}, input_tensor.type() );
        at::Tensor output_tensor = at::zeros_like(input_tensor);
        for(int b=0; b < batch_size; b++){

            auto ref_ptr = &(static_cast<double*>(image_tensor.view(
                image_tensor.numel()
            ).data_ptr())[b * num_super_pixels * n_reference_channels]);
            auto pos_ptr = &(static_cast<double*>(positions.view(
                positions.numel()
            ).data_ptr())[b * num_super_pixels * pd]);
            auto in_ptr = &(static_cast<double*>(input_tensor.view
                (input_tensor.numel()
            ).data_ptr())[b * num_super_pixels * n_input_channels]);
            auto out_ptr = &(static_cast<double*>(output_tensor.view(
                output_tensor.numel()
            ).data_ptr())[b * num_super_pixels * n_input_channels]);

            ComputeKernel<CPUDevice, double>()(
                ref_ptr,
                pos_ptr,
                num_super_pixels,
                n_spatial_dims,
                spatial_dims,
                n_reference_channels,
                spatial_std,
                features_std
            );

            LatticeFilter<CPUDevice, double>()(
                out_ptr,
                in_ptr,
                pos_ptr,
                num_super_pixels,
                pd,
                vd,
                backward
            );
        }
        delete[](spatial_dims);
    return output_tensor;
}


/**
    comput the high dim filter
*/
at::Tensor
LatticeFilter_forward(
    const at::Tensor& input_tensor,
    const at::Tensor& image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0
)
{
    return LatticeFilter_calculate(
        input_tensor,
        image_tensor,
        bilateral,
        theta_alpha,
        theta_beta,
        theta_gamma,
        false // forward
    );
}

at::Tensor
LatticeFilter_backward(
    const at::Tensor& input_tensor,
    const at::Tensor& image_tensor,
    bool bilateral,
    float theta_alpha=1.0,
    float theta_beta=1.0,
    float theta_gamma=1.0
)
{
    return LatticeFilter_calculate(
        input_tensor,
        image_tensor,
        bilateral,
        theta_alpha,
        theta_beta,
        theta_gamma,
        true // backward
    );
}

// bind it to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &LatticeFilter_forward, "LatticeFilter forward");
  m.def("backward", &LatticeFilter_backward, "LatticeFilter backward");
}

// TODO Register the GPU kernels.
