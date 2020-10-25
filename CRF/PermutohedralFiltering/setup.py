from setuptools import setup
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

nvcc_ARCH  = ['-arch=sm_70']
# nvcc_ARCH += ["-gencode=arch=compute_75,code=\"compute_75\""]
# nvcc_ARCH += ["-gencode=arch=compute_75,code=\"sm_75\""]
nvcc_ARCH += ["-gencode=arch=compute_70,code=\"sm_70\""]
# nvcc_ARCH += ["-gencode=arch=compute_61,code=\"sm_61\""]
# nvcc_ARCH += ["-gencode=arch=compute_52,code=\"sm_52\""]
extra_compile_args = { 
            'cxx': ['-Wno-unused-function', '-Wno-write-strings'],
            'nvcc': nvcc_ARCH,}

setup(
    name='Permutohedral',
    ext_modules=[
        CppExtension(
            'Permutohedral',
            ['source/cpu/LatticeFilterKernel.cpp']
        ),
        CUDAExtension(
            'Permutohedral_gpu',
            ['source/gpu/LatticeFilter.cu'],
            # extra_compile_args=extra_compile_args
        )

    ],
    cmdclass={'build_ext': BuildExtension}
)
