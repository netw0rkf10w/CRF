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

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='CRF',
    version='0.0.1',
    author='Đ.Khuê Lê-Huu',
    author_email='khue.le@inria.fr',
    description='Conditional Random Fields for Computer Vision',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/netw0rkf10w/CRF.git',
    project_urls={
        "Bug Tracker": "https://github.com/netw0rkf10w/CRF/issues",
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
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