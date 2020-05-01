/*******************************************************************************************[Vec.h]
Copyright (c) 2003-2007, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
**************************************************************************************************/

#ifndef Minisat_Vec_h
#define Minisat_Vec_h

#include <algorithm>
#include <cassert>
#include <cstring>
#include <new>
#include <type_traits>

#include "minisat/mtl/IntTypes.h"
#include "minisat/mtl/XAlloc.h"

namespace Minisat {

//=================================================================================================
// Automatically resizable arrays
//
// NOTE! Don't use this vector on datatypes that cannot be re-located in memory
// (with realloc)

template <class T>
class vec {
    int m_sz = 0;
    int m_cap = 0;
    T* m_data = nullptr;

    // Helpers for calculating next capacity:
    static inline int imax(int x, int y) {
        int mask = (y - x) >> (sizeof(int) * 8 - 1);
        return (x & mask) + (y & (~mask));
    }

    static inline void nextCap(int& cap) { cap += ((cap >> 1) + 2) & ~1; }

public:
    vec(const vec&) = delete;
    vec(vec&& rhs) noexcept { this->operator=(std::move(rhs)); }

    vec& operator=(const vec&) = delete;
    vec& operator=(vec&& rhs) noexcept {
        std::swap(this->m_data, rhs.m_data);
        std::swap(this->m_sz, rhs.m_sz);
        std::swap(this->m_cap, rhs.m_cap);
        return *this;
    }

    // Constructors:
    vec() = default;
    explicit vec(int size) { growTo(size); }
    vec(int size, const T& pad) { growTo(size, pad); }
    ~vec() { clear(true); }

    // Pointer to first element:
    operator T*(void) { return m_data; }
    T* data() { return m_data; }
    T* begin() { return m_data; }
    T* end() { return m_data + m_sz; }

    // Pointer to first element:
    const T* begin() const { return m_data; }
    const T* end() const { return m_data + m_sz; }

    // Size operations:
    int size() const { return m_sz; }
    void shrink(int nelems) {
        assert(nelems >= 0 && nelems <= m_sz);
        m_sz -= nelems;
    }
    int capacity(void) const { return m_cap; }
    void capacity(int min_cap);
    void growTo(int size);
    void growTo(int size, const T& pad);
    void clear(bool dealloc = false);

    // Stack interface:
    void push(void) {
        if (m_sz == m_cap)
            capacity(m_sz + 1);
        new (&m_data[m_sz]) T();
        m_sz++;
    }
    void push(const T& elem) {
        if (m_sz == m_cap)
            capacity(m_sz + 1);
        m_data[m_sz++] = elem;
    }
    void push_(const T& elem) {
        assert(m_sz < m_cap);
        m_data[m_sz++] = elem;
    }
    void pop() {
        assert(m_sz > 0);
        m_sz--;
    }
    // NOTE: it seems possible that overflow can happen in the 'm_sz+1'
    // expression of 'push()', but in fact it can not since it requires that
    // 'm_cap' is equal to INT_MAX. This in turn can not happen given the way
    // capacities are calculated (below). Essentially, all capacities are even,
    // but INT_MAX is odd.

    const T& last() const { return m_data[m_sz - 1]; }
    T& last() { return m_data[m_sz - 1]; }

    // Vector interface:
    const T& operator[](int index) const { return m_data[index]; }
    T& operator[](int index) { return m_data[index]; }

    // Duplicatation (preferred instead):
    void copyTo(vec<T>& copy) const {
        copy.clear();
        copy.growTo(m_sz);
        ::memcpy(copy.m_data, m_data, sizeof(T) * m_sz);
    }
    void moveTo(vec<T>& dest) {
        dest.clear(true);
        dest.m_data = m_data;
        dest.m_sz = m_sz;
        dest.m_cap = m_cap;
        m_data = NULL;
        m_sz = 0;
        m_cap = 0;
    }
};

template <class T>
void vec<T>::capacity(int min_cap) {
    if (m_cap >= min_cap)
        return;
    int add = imax((min_cap - m_cap + 1) & ~1,
                   ((m_cap >> 1) + 2) & ~1);  // NOTE: grow by approximately 3/2
    if (add <= INT_MAX - m_cap) {
        m_cap += add;
        m_data = static_cast<T*>(::realloc(m_data, m_cap * sizeof(T)));
        if (m_data) {
            return;
        }
    }
    throw OutOfMemoryException{};
}

template <class T>
void vec<T>::growTo(int size, const T& pad) {
    if (m_sz >= size)
        return;
    capacity(size);
    std::fill(m_data + m_sz, m_data + size, pad);
    m_sz = size;
}

template <class T>
void vec<T>::growTo(int size) {
    if (m_sz >= size)
        return;
    capacity(size);
    for (int i = m_sz; i < size; i++) {
        new (&m_data[i]) T();
    }
    m_sz = size;
}

template <class T>
void vec<T>::clear(bool dealloc) {
    static_assert(std::is_trivially_copyable<T>::value);
    if (m_data != NULL) {
        m_sz = 0;
        if (dealloc) {
            free(m_data);
            m_data = NULL;
            m_cap = 0;
        }
    }
}

//=================================================================================================
}  // namespace Minisat

#endif
