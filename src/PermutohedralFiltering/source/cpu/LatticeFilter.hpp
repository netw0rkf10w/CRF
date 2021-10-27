#pragma once

#include "../Devices.hpp"

template <
    typename Device,
    typename T
>
struct LatticeFilter;

template <
    typename T
>
struct
LatticeFilter<
    CPUDevice,
    T
>
{
    void
    operator()
    (
        T* output,
        const T *input,
        const T *positions,
        int num_super_pixels,
        int pd,
        int vd,
        bool reverse
    )
    {
        auto lattice = PermutohedralLatticeCPU<T>(
            pd,
            vd,
            num_super_pixels
        );
        lattice.filter(
            output,
            input,
            positions,
            reverse
        );
    }
};
