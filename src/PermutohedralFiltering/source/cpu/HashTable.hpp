/*Copyright (c) 2018 Miguel Monteiro, Andrew Adams, Jongmin Baek, Abe Davis

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

#pragma once

#include <cstring>
#include <memory>

/***************************************************************/
/* Hash table implementation for permutohedral lattice
 *
 * The lattice points are stored sparsely using a hash table.
 * The key for each point is its spatial location in the (pd+1)-
 * dimensional space.
 */
/***************************************************************/
template <typename T>
class HashTableCPU
{
public:
    short *keys;
    T *values;
    int *entries;
    size_t capacity, filled;
    int pd, vd;

    /* Hash function used in this implementation. A simple base conversion. */
    size_t hash(const short *key) {
        size_t k = 0;
        for (int i = 0; i < pd; i++) {
            k += key[i];
            k *= 2531011;
        }
        return k;
    }

    /* Returns the index into the hash table for a given key.
    *     key: a pointer to the position vector.
    *       h: hash of the position vector.
    *  create: a flag specifying whether an entry should be created,
    *          should an entry with the given key not found.
    */
    int lookupOffset(const short *key, size_t h, bool create = true) {

        // Double hash table size if necessary
        if (filled >= (capacity / 2) - 1) { grow(); }

        // Find the entry with the given key
        while (true) {
            int* e = entries + h;
            // check if the cell is empty
            if (*e == -1) {
                if (!create)
                    return -1; // Return not found.
                // need to create an entry. Store the given key.
                for (int i = 0; i < pd; i++)
                    keys[filled * pd + i] = key[i];
                *e = static_cast<int>(filled);
                filled++;
                return *e * vd;
            }

            // check if the cell has a matching key
            bool match = true;
            for (int i = 0; i < pd && match; i++)
                match = keys[*e*pd + i] == key[i];
            if (match)
                return *e * vd;

            // increment the bucket with wraparound
            h++;
            if (h == capacity)
                h = 0;
        }
    }

    /* Grows the size of the hash table */
    void grow() {

        size_t oldCapacity = capacity;
        capacity *= 2;

        // Migrate the value vectors.
        auto newValues = new T[vd * capacity / 2]{0};
        std::memcpy(newValues, values, sizeof(T) * vd * filled);
        delete[] values;
        values = newValues;

        // Migrate the key vectors.
        auto newKeys = new short[pd * capacity / 2];
        std::memcpy(newKeys, keys, sizeof(short) * pd * filled);
        delete[] keys;
        keys = newKeys;

        auto newEntries = new int[capacity];
        memset(newEntries, -1, capacity*sizeof(int));

        // Migrate the table of indices.
        for (size_t i = 0; i < oldCapacity; i++) {
            if (entries[i] == -1)
                continue;
            size_t h = hash(keys + entries[i] * pd) % capacity;
            while (newEntries[h] != -1) {
                h++;
                if (h == capacity) h = 0;
            }
            newEntries[h] = entries[i];
        }
        delete[] entries;
        entries = newEntries;
    }

public:
    /* Constructor
     *  pd_: the dimensionality of the position vectors on the hyperplane.
     *  vd_: the dimensionality of the value vectors
     */
    HashTableCPU(int pd_, int vd_) : pd(pd_), vd(vd_) {
        capacity = 1 << 15;
        filled = 0;
        entries = new int[capacity];
        memset(entries, -1, capacity*sizeof(int));
        keys = new short[pd * capacity / 2];
        values = new T[vd * capacity / 2]{0};
    }

    ~HashTableCPU(){
        delete[](entries);
        delete[](keys);
        delete[](values);
    }

    // Returns the number of vectors stored.
    int size() { return filled; }

    // Returns a pointer to the keys array.
    short *getKeys() { return keys; }

    // Returns a pointer to the values array.
    T *getValues() { return values; }

    /* Looks up the value vector associated with a given key vector.
     *        k : pointer to the key vector to be looked up.
     *   create : true if a non-existing key should be created.
     */
    T*
    lookup
    (
        short *k,
        bool create = true
    )
    {
        size_t h = hash(k) % capacity;
        int offset = lookupOffset(
            k,
            h,
            create
        );
        if (offset < 0)
            return nullptr;
        else
            return values + offset;
    }
};
