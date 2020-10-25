// see https://github.com/MiguelMonteiro/permutohedral_lattice
#pragma once

#include "DeviceMemoryAllocator.h"


template<
    typename T,
    int pd,
    int vd
>
class HashTableGPU
{
public:
    int capacity;
    T * values;
    short * keys;
    int * entries;

    HashTableGPU(
        int capacity_,
        DeviceMemoryAllocator* allocator
    ):
        capacity(capacity_),
        values(nullptr),
        keys(nullptr),
        entries(nullptr)
    {

        allocator->allocate_device_memory<T>((void**)&values, capacity * vd);
        allocator->memset<T>((void*)values, 0, capacity * vd);

        allocator->allocate_device_memory<int>((void**)&entries, capacity * 2);
        allocator->memset<int>((void*)entries, -1, capacity * 2);

        allocator->allocate_device_memory<short>((void**)&keys, capacity * pd);
        allocator->memset<short>((void*)keys, 0, capacity * pd);
    }

    __device__
    int
    modHash(
        unsigned int n
    )
    {
        return(n % (2 * capacity));
    }

    __device__
    unsigned int
    hash(
        short *key
    )
    {
        unsigned int k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k = k * 2531011;
        }
        return k;
    }

    __device__
    int
    insert(
        short *key,
        unsigned int slot
    )
    {
        int h = modHash(hash(key));
        while (1) {
            int *e = entries + h;

            // If the cell is empty (-1), lock it (-2)
            int contents = atomicCAS(e, -1, -2);

            if (contents == -2){
                // If it was locked already, move on to the next cell
            }else if (contents == -1) {
                // If it was empty, we successfully locked it. Write our key.
                for (int i = 0; i < pd; i++) {
                    keys[slot * pd + i] = key[i];
                }
                // Unlock
                atomicExch(e, slot);
                return h;
            } else {
                // The cell is unlocked and has a key in it, check if it matches
                bool match = true;
                for (int i = 0; i < pd && match; i++) {
                    match = (keys[contents*pd+i] == key[i]);
                }
                if (match)
                    return h;
            }
            // increment the bucket with wraparound
            h++;
            if (h == capacity*2)
                h = 0;
        }
    }

    __device__
    int
    retrieve(
        short *key
    )
    {

        int h = modHash(hash(key));
        while (1) {
            int *e = entries + h;

            if (*e == -1)
                return -1;

            bool match = true;
            for (int i = 0; i < pd && match; i++) {
                match = (keys[(*e)*pd+i] == key[i]);
            }
            if (match)
                return *e;

            h++;
            if (h == capacity*2)
                h = 0;
        }
    }
};