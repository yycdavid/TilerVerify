/****************************  [utils/Random.h] ****************************
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson
Copyright (c) 2020, Kai Jia

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
***************************************************************************/

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>

namespace Minisat {
// see
// https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/
class RandomState {
    static_assert(sizeof(size_t) == 8,
                  "the RNG is only optimized for 64-bit platforms");
    static constexpr double DMAX = std::numeric_limits<uint64_t>::max() + 1.0;
    __uint128_t m_state;

    uint64_t next() {
        m_state *= 0xda942042e4dd58b5ull;
        return m_state >> 64;
    }

public:
    explicit RandomState(uint64_t seed) {
        std::mt19937_64 rng;
        rng.seed(seed);
        m_state = (static_cast<__uint128_t>(rng()) << 64) | rng();
    }

    //! a uniform real in [0, 1)
    double uniform() { return next() / DMAX; }

    //! return true with probability p
    bool binomial(double p) { return next() < (DMAX * p); }

    //! random value in [0, upper-1]
    int randint(int upper) {
        return std::min<int>(uniform() * upper, upper - 1);
    }
};
}  // namespace Minisat

