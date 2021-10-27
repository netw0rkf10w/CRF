import setuptools
from torch.utils.cpp_extension import CppExtension, BuildExtension, CUDAExtension

import site
site.ENABLE_USER_SITE = True

# setuptools.setup()

nvcc_ARCH  = ['-arch=sm_70']
# nvcc_ARCH += ["-gencode=arch=compute_75,code=\"compute_75\""]
# nvcc_ARCH += ["-gencode=arch=compute_75,code=\"sm_75\""]
nvcc_ARCH += ["-gencode=arch=compute_70,code=\"sm_70\""]
# nvcc_ARCH += ["-gencode=arch=compute_61,code=\"sm_61\""]
# nvcc_ARCH += ["-gencode=arch=compute_52,code=\"sm_52\""]
extra_compile_args = { 
            'cxx': ['-Wno-unused-function', '-Wno-write-strings'],
            'nvcc': nvcc_ARCH,}

setuptools.setup(name='CRF',
    ext_modules=[
            CppExtension(
                'Permutohedral',
                ['src/PermutohedralFiltering/source/cpu/LatticeFilterKernel.cpp']
            ),
            CUDAExtension(
                'Permutohedral_gpu',
                ['src/PermutohedralFiltering/source/gpu/LatticeFilter.cu'],
                extra_compile_args=extra_compile_args
            )
        ],
      cmdclass={'build_ext': BuildExtension}
      )