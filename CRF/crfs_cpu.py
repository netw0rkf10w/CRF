"""
Python wrapper for DenseCRF
D. Khue Le-Huu
"""
import os
import numpy as np
from ctypes import *

def crf_inference():
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DenseCRF', 'libdensecrf.so')
    libc = CDLL(lib_path)
    ADGM = libc.ADGM_C4


def adgm_C4(affinityMatrix, n1, n2,
         rho_min=0.000001, rho_max=100,
         iter1=200, iter2=50, step=2.0,
         max_iter=10000, precision=1e-5,
         verbose=True):
    """
    Alternating direction graph matching: https://github.com/netw0rkf10w/adgm
    """
    # sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adgm'))
    # libc = cdll.LoadLibrary("adgm/libADGMv2.so")
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libadgm.so')
    # libc = CDLL("/home/ubuntu/khue/shelf-analyzer/src/adgm/libADGMv2.so")
    # libc = CDLL(lib_path, RTLD_GLOBAL)
    libc = CDLL(lib_path)
    ADGM = libc.ADGM_C4

    # print(libc)
    N = n1*n2
    c_M = (c_double * (N*N))(*(affinityMatrix.flatten()))
    # print(c_M)

    # x = (c_double * N)()
    # energy = libc.ADGM_C(byref(x), byref(energies), byref(residuals),
    #           c_int(n1), c_int(n2), c_M,
    #           c_double(rho_min), c_double(rho_max), c_int(iter1), c_int(iter2),
    #           c_double(step), c_int(MAX_ITER), c_double(precision), c_bool(verbose))

    
    ADGM.argtypes = ()
    ADGM.restype = POINTER(c_double)
    ADGM.argtypes = [POINTER(c_double), c_int, c_int,
                        c_double, c_double, c_int, c_int,
                        c_double, c_int, c_double, c_bool]

    # ADGM_C2.restype = POINTER(c_int)

    # from numpy.ctypeslib import ndpointer
    # ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # if sys.version_info[0] >= 3:
    #     ADGM_C2.restype = POINTER(c_double)
    # else:
    #     # ValueError: '<P' is not a valid PEP 3118 buffer format string
    #     from numpy.ctypeslib import ndpointer
    #     ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # important: switching n1, n2 because the matrix in C++ code n2*n1
    x = ADGM(c_M, c_int(n2), c_int(n1),
                c_double(rho_min), c_double(rho_max),
                c_int(iter1), c_int(iter2),
                c_double(step), c_int(max_iter),
                c_double(precision), c_bool(verbose))

    X = [x[i] for i in range(N)]
    X = np.asarray(X)
    energy = np.dot(np.dot(X.T, affinityMatrix), X)
    print('energy =', energy)
    X = X.reshape((n1, n2))
    # print(X)
    return X




def adgm_C2(affinityMatrix, n1, n2,
         rho_min=0.001, rho_max=100,
         iter1=200, iter2=50, step=2.0,
         max_iter=10000, precision=1e-5,
         verbose=True):
    """
    Alternating direction graph matching: https://github.com/netw0rkf10w/adgm
    """
    # sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adgm'))
    # libc = cdll.LoadLibrary("adgm/libADGMv2.so")
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libadgm.so')
    # libc = CDLL("/home/ubuntu/khue/shelf-analyzer/src/adgm/libADGMv2.so")
    # libc = CDLL(lib_path, RTLD_GLOBAL)
    libc = CDLL(lib_path)
    # print(libc)
    N = n1*n2
    c_M = (c_double * (N*N))(*(affinityMatrix.flatten()))
    # print(c_M)
    
    _residuals = POINTER(c_double)()
    n_residuals = c_int()

    # x = (c_double * N)()
    # energy = libc.ADGM_C(byref(x), byref(energies), byref(residuals),
    #           c_int(n1), c_int(n2), c_M,
    #           c_double(rho_min), c_double(rho_max), c_int(iter1), c_int(iter2),
    #           c_double(step), c_int(MAX_ITER), c_double(precision), c_bool(verbose))

    ADGM_C2 = libc.ADGM_C2
    ADGM_C2.argtypes = ()
    ADGM_C2.restype = POINTER(c_int)
    # ADGM_C2.argtypes = [POINTER(c_double), POINTER(c_int),
    #                     POINTER(c_double),c_int, c_int,
    #                     c_double, c_double, c_int, c_int,
    #                     c_double, c_int, c_double, c_bool]

    # ADGM_C2.restype = POINTER(c_int)

    # from numpy.ctypeslib import ndpointer
    # ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # if sys.version_info[0] >= 3:
    #     ADGM_C2.restype = POINTER(c_double)
    # else:
    #     # ValueError: '<P' is not a valid PEP 3118 buffer format string
    #     from numpy.ctypeslib import ndpointer
    #     ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # important: switching n1, n2 because the matrix in C++ code n2*n1
    x = ADGM_C2(_residuals, byref(n_residuals),
                c_M, c_int(n2), c_int(n1),
                c_double(rho_min), c_double(rho_max),
                c_int(iter1), c_int(iter2),
                c_double(step), c_int(max_iter),
                c_double(precision), c_bool(verbose))
    #
    nr = n_residuals.value
    print('nr =', nr)
    # residuals = np.fromiter(_residuals, dtype=np.float, count=nr)
    # energies = np.fromiter(_energies, dtype=np.float, count=ne)
    residuals = [_residuals[i] for i in range(nr)]
    print('residuals =', residuals)

    # print(X)


    X = [x[i] for i in range(N)]
    X = np.asarray(X)
    energy = np.dot(np.dot(X.T, affinityMatrix), X)
    print('energy =', energy)
    X = X.reshape((n1, n2))
    # print(X)
    return X



def adgm_C3(affinityMatrix, n1, n2,
         rho_min=0.001, rho_max=100,
         iter1=200, iter2=50, step=2.0,
         max_iter=10000, precision=1e-5,
         verbose=True):
    """
    Alternating direction graph matching: https://github.com/netw0rkf10w/adgm
    """
    # sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'adgm'))
    # libc = cdll.LoadLibrary("adgm/libADGMv2.so")
    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'libadgm.so')
    # libc = CDLL("/home/ubuntu/khue/shelf-analyzer/src/adgm/libADGMv2.so")
    # libc = CDLL(lib_path, RTLD_GLOBAL)
    libc = CDLL(lib_path)
    ADGM = libc.ADGM_C3

    # print(libc)
    N = n1*n2
    c_M = (c_double * (N*N))(*(affinityMatrix.flatten()))
    # print(c_M)
    
    _residuals = POINTER(POINTER(c_double))()
    n_residuals = POINTER(c_int)()

    # x = (c_double * N)()
    # energy = libc.ADGM_C(byref(x), byref(energies), byref(residuals),
    #           c_int(n1), c_int(n2), c_M,
    #           c_double(rho_min), c_double(rho_max), c_int(iter1), c_int(iter2),
    #           c_double(step), c_int(MAX_ITER), c_double(precision), c_bool(verbose))

    
    # ADGM_C2.argtypes = ()
    ADGM.restype = POINTER(c_double)
    ADGM.argtypes = [POINTER(POINTER(c_double)), POINTER(c_int),
                        POINTER(c_double),c_int, c_int,
                        c_double, c_double, c_int, c_int,
                        c_double, c_int, c_double, c_bool]

    # ADGM_C2.restype = POINTER(c_int)

    # from numpy.ctypeslib import ndpointer
    # ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # if sys.version_info[0] >= 3:
    #     ADGM_C2.restype = POINTER(c_double)
    # else:
    #     # ValueError: '<P' is not a valid PEP 3118 buffer format string
    #     from numpy.ctypeslib import ndpointer
    #     ADGM_C2.restype = ndpointer(dtype=c_double, shape=(N,))

    # important: switching n1, n2 because the matrix in C++ code n2*n1
    x = ADGM(_residuals, n_residuals,
                c_M, c_int(n2), c_int(n1),
                c_double(rho_min), c_double(rho_max),
                c_int(iter1), c_int(iter2),
                c_double(step), c_int(max_iter),
                c_double(precision), c_bool(verbose))
    #
    nr = n_residuals.value
    print('nr =', nr)
    # residuals = np.fromiter(_residuals, dtype=np.float, count=nr)
    # energies = np.fromiter(_energies, dtype=np.float, count=ne)
    residuals = [_residuals[i] for i in range(nr)]
    print('residuals =', residuals)

    # print(X)


    X = [x[i] for i in range(N)]
    X = np.asarray(X)
    energy = np.dot(np.dot(X.T, affinityMatrix), X)
    print('energy =', energy)
    X = X.reshape((n1, n2))
    # print(X)
    return X


def test_adgm():
    n1 = 4
    n2 = 6
    N = n1*n2
    M = np.random.rand(N, N)
    # np.save('M.npy', M)
    # M = np.load('M.npy')
    X = adgm(M, n1, n2, verbose=True)
    print('sum over col')
    print(np.sum(X, axis=0))
    print('sum over row')
    print(np.sum(X, axis=1))

if __name__ == "__main__":
    test_adgm()
