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

//#include "LatticeFilterKernel.h"

#include <torch/extension.h>

#include <iostream>

#include "PermutohedralLatticeGPU.cuh"
#include "DeviceMemoryAllocator.h"
#include "../Devices.hpp"
#ifndef SPATIAL_DIMS
#define SPATIAL_DIMS 2
#endif
#ifndef INPUT_CHANNELS
#define INPUT_CHANNELS 3
#endif
#ifndef REFERENCE_CHANNELS
#define REFERENCE_CHANNELS 3
#endif
#define CHECK_CUDA(x) AT_ASSERT(x.type().is_cuda())
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous())
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


void computeKernelGPU(
    const float* input_image,
    float* positions,
    int num_super_pixels,
    int n_spatial_dims,
    int *spatial_dims,
    int n_reference_channels,
    float spatial_std,
    float features_std
)
{

    // printf("spatial_std = %f, n_spatial_dims = %d, spatial_dims[0] = %d, spatial_dims[1] = %d\n",
    //         spatial_std, n_spatial_dims, spatial_dims[0], spatial_dims[1]);

    int* spatial_dims_gpu;
    
    gpuErrchk(
        cudaMalloc(
            (void**)&spatial_dims_gpu,
            n_spatial_dims*sizeof(int)
        )
    );

    gpuErrchk(
        cudaMemcpy(
            spatial_dims_gpu,
            spatial_dims,
            n_spatial_dims*sizeof(int),
            cudaMemcpyHostToDevice
        )
    );

    // auto allocator = DeviceMemoryAllocator(context);
    // allocator.allocate_device_memory<int>((void**)&spatial_dims_gpu, n_spatial_dims);
    // gpuErrchk(cudaMemcpy(spatial_dims_gpu, spatial_dims, n_spatial_dims*sizeof(int), cudaMemcpyHostToDevice));

    // // DEBUG: Retrieves data from device and print
    // int* spatial_dims_copy = (int*) malloc(n_spatial_dims*sizeof(int));
    // // cudaMalloc((void**)&spatial_dims_copy, n_spatial_dims*sizeof(int));
    // cudaMemcpy(spatial_dims_copy, spatial_dims_gpu, n_spatial_dims*sizeof(int), cudaMemcpyDeviceToHost);
    // printf("spatial_dims_copy[0] = %d, spatial_dims_copy[1] = %d\n", spatial_dims_copy[0], spatial_dims_copy[1]);

    dim3 blocks((num_super_pixels - 1) / BLOCK_SIZE + 1, 1, 1);
    dim3 blockSize(BLOCK_SIZE, 1, 1);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    //return input_tensor;

    // std::cout<<"positions before: \n"<<positions<<std::endl;
    // std::cout<<"image: \n"<<input_image<<std::endl;

    compute_kernel<float><<<blocks, blockSize>>>(
        input_image,
        positions,
        num_super_pixels,
        n_reference_channels,
        n_spatial_dims,
        spatial_dims_gpu,
        spatial_std,
        features_std
    );

    // std::cout<<"positions after: \n"<<positions<<std::endl;

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    gpuErrchk(
        cudaFree(
            spatial_dims_gpu
        )
    );
};

//declaration of what lattices (pd and vd) can be used


// Define the GPU implementation that launches the CUDA kernel.
template<int pd,int vd> void latticeFilterGPU(
    float* output,
    const float* input,
    const float* positions,
    int num_super_pixels,
    bool reverse)
{
    auto allocator = DeviceMemoryAllocator();
    //bilateral

    auto lattice = PermutohedralLatticeGPU<float,pd,vd>(num_super_pixels, &allocator);

    lattice.filter(output, input, positions,reverse);
}

at::Tensor LatticeFilter_calculate_gpu(at::Tensor const & input_tensor,
                                       at::Tensor const & image_tensor,
                                       bool bilateral,
                                       float theta_alpha=1.0,
                                       float theta_beta=1.0,
                                       float theta_gamma=1.0,
                                       bool backward=false,
                                       bool nhwc=true)
{
    
    int rank = image_tensor.ndimension();
    int n_spatial_dims = rank - 2;
    int pd;

    // dimension 0 is batch
    auto batch_size = static_cast<int>(input_tensor.size(0));

    int channel_dim;
    if (nhwc) {
        channel_dim = -1;
    } else {
        channel_dim = 1;
    }

    auto n_input_channels = static_cast<int>(input_tensor.size(channel_dim));
    
    auto spatial_dims = new int[n_spatial_dims];

    int num_super_pixels{1};
    int dim_idx;
    for (int i = 0; i < n_spatial_dims; i++){
        if (nhwc) {
            dim_idx = i + 1; // spatial dimension starts after batch dimension
        } else {
            dim_idx = i + 2; // spatial dimension starts after batch dimension and channel dimension
        }
        auto dim_size = static_cast<int>(image_tensor.size(dim_idx));
        num_super_pixels *= dim_size;
        spatial_dims[i] = dim_size;
    }


    int vd = n_input_channels + 1;
    float spatial_std;
    float features_std;
    int n_reference_channels;

    if(bilateral){
        assert(image_tensor.dim() == rank);
        n_reference_channels = static_cast<int>(image_tensor.size(channel_dim));
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

    at::Tensor positions = at::zeros(
        {batch_size * num_super_pixels * pd},
        input_tensor.type()
    );
    at::Tensor output_tensor = at::zeros_like(input_tensor);

    CHECK_INPUT(image_tensor);
    CHECK_INPUT(positions);

    for(int b=0; b < batch_size; b++){

        auto ref_ptr = &(image_tensor.data<float>()[b * num_super_pixels * n_reference_channels]);
        auto pos_ptr = &(positions.data<float>()[b * num_super_pixels * pd]);
        auto in_ptr = &(input_tensor.data<float>()[b * num_super_pixels * n_input_channels]);
        auto out_ptr = &(output_tensor.data<float>()[b * num_super_pixels * n_input_channels]);

        computeKernelGPU(ref_ptr, pos_ptr,
                    num_super_pixels, n_spatial_dims, spatial_dims,
                    n_reference_channels, spatial_std, features_std);

        
        // std::cout<<"input tensor: \n"<<input_tensor<<std::endl;
        // std::cout<<"positions: \n"<<positions<<std::endl;
        // printf("pd = %d, vd = %d\n", pd, vd);
    
        if(pd==2 and vd==27)
        {
            latticeFilterGPU<2,27>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==27)
        {
            latticeFilterGPU<5,27>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==4)
        {
            latticeFilterGPU<2,4>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==4)
        {
            latticeFilterGPU<5,4>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==35)
        {
            latticeFilterGPU<2,35>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==35)
        {
            latticeFilterGPU<5,35>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==6)
        {
            latticeFilterGPU<2,6>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==3 and vd==6)
        {
            latticeFilterGPU<3,6>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==6)
        {
            latticeFilterGPU<5,6>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==22)
        {
            latticeFilterGPU<2,22>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==22)
        {
            latticeFilterGPU<5,22>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==3)
        {
            latticeFilterGPU<2,3>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==3)
        {
            latticeFilterGPU<5,3>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==2 and vd==46)
        {
            latticeFilterGPU<2,46>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else if(pd==5 and vd==46)
        {
            latticeFilterGPU<5,46>(out_ptr, in_ptr, pos_ptr, num_super_pixels, backward);
        }
        else
        {
                /**
                Sorry that you need to do this. But it is quite simple. You need to add an additional litticeFilterGpu.
                1. notice the pd and vd value from the error message
                2. add an else if(pd == ? and vd == ?)
                3. copy the following text. Replace vd and pd with the values
                    latticeFilterGPU<pd,vd>(output_tensor, input_tensor,
                        positions, num_super_pixels, backward);
                4. recomplie  with setup.py
                */
            std::cerr << "latticeFilterGPU with pd=" << pd << " and vd=" << vd << " doesnt exists. Pls add this in"
                        << " the file latticeFilter.cu. This is nessacary for an efficent GPU implementation" <<std::endl;
            exit(1);
        }
    }

    // std::cout<<"output_tensor: \n"<<output_tensor<<std::endl;
    delete[](spatial_dims);
    return output_tensor;
}



at::Tensor LatticeFilter_forward(at::Tensor const & input_tensor,
                                at::Tensor const & image_tensor,
                                bool bilateral,
                                float theta_alpha=1.0,
                                float theta_beta=1.0,
                                float theta_gamma=1.0)
{
    return LatticeFilter_calculate_gpu(input_tensor, image_tensor, bilateral,
                            theta_alpha, theta_beta, theta_gamma, false);
}

at::Tensor LatticeFilter_backward(at::Tensor const & input_tensor,
                                  at::Tensor const & image_tensor,
                                  bool bilateral,
                                  float theta_alpha=1.0,
                                  float theta_beta=1.0,
                                  float theta_gamma=1.0)
{
    return LatticeFilter_calculate_gpu(input_tensor, image_tensor, bilateral,
        theta_alpha, theta_beta, theta_gamma, true);
}

// bind it to python
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &LatticeFilter_forward, "LatticeFilter forward");
  m.def("backward", &LatticeFilter_backward, "LatticeFilter backward");
}
