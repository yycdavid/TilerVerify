/***********************************************************************************[SolverTypes.h]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
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

#ifndef Minisat_SolverTypes_h
#define Minisat_SolverTypes_h

#include <cassert>
#include <utility>
#include <vector>

#include "minisat/mtl/Alg.h"
#include "minisat/mtl/Alloc.h"
#include "minisat/mtl/IntTypes.h"
#include "minisat/mtl/Vec.h"

namespace Minisat {

//=================================================================================================
// Variables, literals, lifted booleans, clauses:

// NOTE! Variables are just integers. No abstraction here. They should be chosen
// from 0..N, so that they can be used as array indices.

typedef int Var;
#define var_Undef (-1)

struct Lit {
    int x;

    bool operator==(Lit p) const { return x == p.x; }
    bool operator!=(Lit p) const { return x != p.x; }
    bool operator<(Lit p) const {
        return x < p.x;
    }  // '<' makes p, ~p adjacent in the ordering.
};

static inline Lit mkLit(Var var, bool sign = false) {
    Lit p;
    p.x = var + var + (int)sign;
    return p;
}
static inline Lit operator~(Lit p) {
    Lit q;
    q.x = p.x ^ 1;
    return q;
}
static inline Lit operator^(Lit p, bool b) {
    Lit q;
    q.x = p.x ^ (unsigned int)b;
    return q;
}
static inline bool sign(Lit p) {
    return p.x & 1;
}
static inline int var(Lit p) {
    return p.x >> 1;
}

// Mapping Literals to and from compact integers suitable for array indexing:
static inline int toInt(Var v) {
    return v;
}
static inline int toInt(Lit p) {
    return p.x;
}
static inline Lit toLit(int i) {
    Lit p;
    p.x = i;
    return p;
}

// const Lit lit_Undef = mkLit(var_Undef, false);  // }- Useful special
// constants. const Lit lit_Error = mkLit(var_Undef, true );  // }

static constexpr Lit lit_Undef = {-2};  // }- Useful special constants.
static constexpr Lit lit_Error = {-1};  // }

//=================================================================================================
// Lifted booleans:
//
// NOTE: this implementation is optimized for the case when comparisons between
// values are mostly between one variable and one constant. Some care had to be
// taken to make sure that gcc does enough constant propagation to produce
// sensible code, and this appears to be somewhat fragile unfortunately.

struct lbool_true_type {};
struct lbool_false_type {};
struct lbool_undef_type {};
struct lbool_invalid_type {};
static constexpr lbool_true_type l_True;
static constexpr lbool_false_type l_False;
static constexpr lbool_undef_type l_Undef;
static constexpr lbool_invalid_type l_Invalid;

class lbool {
    uint8_t value;

    explicit constexpr lbool(uint8_t v) : value{v} {}

public:
    constexpr lbool() : value(0) {}
    constexpr explicit lbool(bool v) : value{!v} {}
    constexpr lbool(lbool_true_type) : lbool(true) {}
    constexpr lbool(lbool_false_type) : lbool(false) {}
    constexpr lbool(lbool_undef_type) : lbool(static_cast<uint8_t>(2)) {}
    constexpr lbool(lbool_invalid_type) : lbool(static_cast<uint8_t>(4)) {}

    template <typename T>
    lbool(T) = delete;

    static constexpr lbool make_raw(uint8_t v) { return lbool{v}; }

    constexpr bool operator==(lbool_true_type) const { return value == 0; }
    constexpr bool operator==(lbool_false_type) const { return value == 1; }
    constexpr bool operator==(lbool_undef_type) const { return value & 2; }
    constexpr bool operator==(lbool_invalid_type) const { return value & 4; }
    template <typename T>
    constexpr bool operator!=(T t) const {
        return !operator==(t);
    }

    constexpr lbool operator^(bool b) const {
        return make_raw(value ^ (uint8_t)b);
    }

    constexpr bool is_boolv(bool b) const {
        return value == (static_cast<uint8_t>(b) ^ 1);
    }

    template<bool b>
    constexpr bool is_bool() const {
        return value == (static_cast<uint8_t>(b) ^ 1);
    }

    bool as_bool() const {
        assert(!(value & 6));
        return value == 0;
    }

    constexpr bool is_not_undef() const { return !(value & 2); }
};

/* ========================== LeqStatus =========================== */
//! Clause reference type
using CRef = RegionAllocator<uint32_t>::Ref;

//! status for an LEQ clause
union LeqStatus {
    static constexpr uint32_t OFFSET_IN_CLAUSE = 3;

    enum ImplyType {
        //! no var has been implied from this clause
        IMPLY_NONE = 0,
        //! value of dst is implied (preconditions are placed at the beginning)
        IMPLY_DST = 1,
        //! some lits are implied (preconditions are placed at the beginning)
        IMPLY_LITS = 2,
        //! dst var and know lits cause a conflict
        IMPLY_CONFL = 3,
    };

    struct {
        //! number of known true lits
        uint16_t nr_true : 15;
        //! whether preconditions for the imply is true
        uint16_t precond_is_true : 1;
        //! number of decided lits (sum of nr_true and nr_false)
        uint16_t nr_decided : 14;
        //! one value in ImplyType
        uint16_t imply_type : 2;
    };
    uint32_t val_u32;

    void decr(uint32_t delta_nr_true, uint32_t delta_nr_decided) {
        val_u32 -= (delta_nr_decided << 16) | delta_nr_true;
    };

    void incr(uint32_t delta_nr_true, uint32_t delta_nr_decided) {
        val_u32 += (delta_nr_decided << 16) | delta_nr_true;
    };

    //! if bit = 1, set imply_type to 0; otherwise it is unchanged.
    //! bit should be either 0 or 1
    void clear_imply_type_with(uint32_t bit) {
        val_u32 &= ((1u << 30) - 1) | (bit - 1);
    }

    //! get CRef of rellocated clause from this var. Note that the cref is a
    //! clause, not for a var
    CRef get_cref_after_reloc() { return *(reinterpret_cast<CRef*>(this) - 3); }

    bool operator==(const LeqStatus& rhs) const {
        return val_u32 == rhs.val_u32;
    }
};
/* ========================== end LeqStatus =========================== */

//=================================================================================================
// Clause -- a simple class for representing a clause:

class Clause {
    /*!
     * Note:
     * 1. learnt and is_leq can not both be true
     * 2. use_extra and is_leq can not both be true
     * 3. If is_leq is true, there would be two extra data items: one is
     *    leq_dst, the other is leq_bound
     * 4. Layout for LEQ clauses: header, lits[], dst, bound, status
     */
    struct {
        unsigned mark : 2;
        unsigned learnt : 1;
        unsigned is_leq : 1;
        unsigned has_extra : 1;
        unsigned reloced : 1;
        unsigned size : 26;
    } header;
    union Data {
        Lit lit;
        float act;
        uint32_t abs;
        int32_t leq_bound;
        LeqStatus leq_status;
        CRef rel;
    };
    Data data[0];

    friend class ClauseAllocator;

    // NOTE: This constructor cannot be used directly (doesn't allocate enough
    // memory).
    template <class V>
    Clause(const V& ps, bool use_extra, bool learnt, bool is_leq) {
        assert(!learnt || !is_leq);
        assert(!use_extra || !is_leq);
        assert(ps.size() < (1 << 26));
        // leq size determined by LeqStatus and LeqWatcher
        assert(!is_leq || ps.size() < (1 << 14));

        header.mark = 0;
        header.learnt = learnt;
        header.is_leq = is_leq;
        header.has_extra = use_extra;
        header.reloced = 0;
        header.size = ps.size();

        for (int i = 0; i < ps.size(); i++) {
            data[i].lit = ps[i];
        }

        if (header.has_extra) {
            if (header.learnt)
                data[header.size].act = 0;
            else
                calcAbstraction();
        }
    }

public:
    void calcAbstraction() {
        assert(header.has_extra);
        uint32_t abstraction = 0;
        for (int i = 0; i < size(); i++)
            abstraction |= 1u << (var(data[i].lit) & 31);
        data[header.size].abs = abstraction;
    }

    int size() const { return header.size; }
    void shrink(int i) {
        assert(i <= size() && !header.is_leq);
        if (header.has_extra)
            data[header.size - i] = data[header.size];
        header.size -= i;
    }
    void pop() { shrink(1); }

    void shrink_leq_to(int new_size, int new_bound) {
        assert(header.is_leq && !header.has_extra);
        data[new_size] = data[header.size];
        data[new_size + 1].leq_bound = new_bound;
        data[new_size + 2] = data[header.size + 2];
        header.size = new_size;
    }

    bool learnt() const { return header.learnt; }
    bool is_leq() const { return header.is_leq; }
    Lit leq_dst() const { return data[header.size].lit; }
    int leq_bound() const { return data[header.size + 1].leq_bound; }
    LeqStatus& leq_status() { return data[header.size + 2].leq_status; }
    LeqStatus leq_status() const { return data[header.size + 2].leq_status; }
    bool has_extra() const { return header.has_extra; }
    uint32_t mark() const { return header.mark; }
    void mark(uint32_t m) { header.mark = m; }
    Lit last() const { return data[header.size - 1].lit; }

    const Lit* lit_data() const {
        static_assert(sizeof(Data) == sizeof(Lit));
        return &data[0].lit;
    }

    bool reloced() const { return header.reloced; }
    CRef relocation() const { return data[0].rel; }

    //! relocate to another clause; this can only be called once and is
    //! destructive
    void record_relocate(CRef c) {
        assert(!header.reloced);
        header.reloced = 1;
        data[0].rel = c;
        // for LEQ clauses
        data[size() - 1].rel = c;
    }

    // NOTE: somewhat unsafe to change the clause in-place! Must manually call
    // 'calcAbstraction' afterwards for
    //       subsumption operations to behave correctly.
    Lit& operator[](int i) { return data[i].lit; }
    Lit operator[](int i) const { return data[i].lit; }
    operator const Lit*(void)const { return (Lit*)data; }

    float& activity() {
        assert(header.has_extra);
        return data[header.size].act;
    }
    uint32_t abstraction() const {
        assert(header.has_extra);
        return data[header.size].abs;
    }

    Lit subsumes(const Clause& other) const;
    void strengthen(Lit p);
};

//=================================================================================================
// ClauseAllocator -- a simple class for allocating memory for clauses:

static constexpr CRef CRef_Undef = RegionAllocator<uint32_t>::Ref_Undef;
class ClauseAllocator : public RegionAllocator<uint32_t> {
    using Super = RegionAllocator<uint32_t>;

    static int clauseWord32Size(int size, bool has_extra, bool is_leq) {
        assert(!has_extra || !is_leq);
        size += static_cast<int>(has_extra) + static_cast<int>(is_leq) * 3;
        return (sizeof(Clause) + (sizeof(Lit) * size)) / sizeof(uint32_t);
    }

public:
    bool extra_clause_field = false;

    ClauseAllocator(uint32_t start_cap) : Super(start_cap) {}
    ClauseAllocator() = default;

    void moveTo(ClauseAllocator& to) {
        to.extra_clause_field = extra_clause_field;
        Super::moveTo(to);
    }

    template <class Lits>
    CRef alloc(const Lits& ps, bool learnt = false, Lit leq_dst = lit_Undef,
               int leq_bound = 0) {
        static_assert(sizeof(Clause::Data) == sizeof(uint32_t));
        bool use_extra = learnt | extra_clause_field;
        bool is_leq = leq_dst != lit_Undef;

        CRef cid = Super::alloc(clauseWord32Size(ps.size(), use_extra, is_leq));
        Clause* cl = new (lea(cid)) Clause{ps, use_extra, learnt, is_leq};

        if (is_leq) {
            cl->data[ps.size()].lit = leq_dst;
            cl->data[ps.size() + 1].leq_bound = leq_bound;
            cl->data[ps.size() + 2].leq_status.val_u32 = 0;
        }

        return cid;
    }

    // Deref, Load Effective Address (LEA), Inverse of LEA (AEL):

    Clause& operator[](Ref r) {
        return reinterpret_cast<Clause&>(Super::operator[](r));
    }
    const Clause& operator[](Ref r) const {
        return const_cast<ClauseAllocator*>(this)->operator[](r);
    }
    Clause* lea(Ref r) { return reinterpret_cast<Clause*>(Super::lea(r)); }
    const Clause* lea(Ref r) const {
        return const_cast<ClauseAllocator*>(this)->lea(r);
    }
    template <typename T>
    T* lea_as(Ref r) {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0 &&
                      alignof(T) % alignof(uint32_t) == 0);
        return reinterpret_cast<T*>(Super::lea(r));
    }
    template <typename T>
    const T* lea_as(Ref r) const {
        return const_cast<ClauseAllocator*>(this)->lea_as<T>(r);
    }
    template <typename T>
    Ref ael(const T* t) {
        static_assert(sizeof(T) % sizeof(uint32_t) == 0 &&
                      alignof(T) % alignof(uint32_t) == 0);
        return Super::ael(reinterpret_cast<const uint32_t*>(t));
    }

    void free(CRef cid) {
        Clause& c = operator[](cid);
        Super::free(clauseWord32Size(c.size(), c.has_extra(), c.is_leq()));
    }

    void reloc(CRef& cr, ClauseAllocator& to) {
        Clause& c = operator[](cr);

        if (c.reloced()) {
            cr = c.relocation();
            return;
        }

        if (c.is_leq()) {
            cr = to.alloc(c, c.learnt(), c.leq_dst(), c.leq_bound());
            to[cr].leq_status() = c.leq_status();
        } else {
            cr = to.alloc(c, c.learnt());
        }
        c.record_relocate(cr);

        // Copy extra data-fields:
        // (This could be cleaned-up. Generalize Clause-constructor to be
        // applicable here instead?)
        to[cr].mark(c.mark());
        if (to[cr].learnt())
            to[cr].activity() = c.activity();
        else if (to[cr].has_extra())
            to[cr].calcAbstraction();
    }
};

//=================================================================================================
/*!
 * a class for maintaining occurence lists with lazy deletion
 *
 * \tparam Refresh: a functor class to refresh the watchers; it can modify
 *      watchers in place. Return true if a watcher should be removed.
 */
template <class Idx, class Vec, class Refresh>
class OccLists {
    std::vector<Vec> m_occs;
    vec<bool> m_dirty;
    vec<Idx> m_dirties;
    Refresh m_refresh;

    //! refresh dirty watches and remove deleted items in the list of idx
    void clean(const Idx& idx);

public:
    template <typename... Args>
    explicit OccLists(Args&&... args)
            : m_refresh{std::forward<Args>(args)...} {}

    void init(const Idx& idx) {
        size_t size = toInt(idx) + 1;
        if (size > m_occs.size()) {
            while (m_occs.size() < size) {
                m_occs.emplace_back();
            }
            m_dirty.growTo(size);
        }
    }

    // Vec&  operator[](const Idx& idx){ return occs[toInt(idx)]; }
    Vec& operator[](const Idx& idx) { return m_occs[toInt(idx)]; }
    Vec& lookup(const Idx& idx) {
        if (m_dirty[toInt(idx)])
            clean(idx);
        return m_occs[toInt(idx)];
    }

    void cleanAll();

    //! mark that watchers in a given index should be refreshed
    void smudge(const Idx& idx) {
        if (m_dirty[toInt(idx)] == 0) {
            m_dirty[toInt(idx)] = 1;
            m_dirties.push(idx);
        }
    }

    void clear(bool free = true) {
        if (free) {
            std::vector<Vec> t;
            m_occs.swap(t);
        } else {
            m_occs.clear();
        }
        m_dirty.clear(free);
        m_dirties.clear(free);
    }
};

template <class Idx, class Vec, class Refresh>
void OccLists<Idx, Vec, Refresh>::cleanAll() {
    for (Idx i : m_dirties) {
        // Dirties may contain duplicates so check here if a variable is already
        // cleaned:
        if (m_dirty[toInt(i)]) {
            clean(i);
        }
    }
    m_dirties.clear();
}

template <class Idx, class Vec, class Refresh>
void OccLists<Idx, Vec, Refresh>::clean(const Idx& idx) {
    Vec& vec = m_occs[toInt(idx)];
    int i, j;
    for (i = j = 0; i < vec.size(); i++) {
        if (!m_refresh(vec[i])) {
            vec[j++] = vec[i];
        }
    }
    vec.shrink(i - j);
    m_dirty[toInt(idx)] = 0;
}

/*_________________________________________________________________________________________________
|
|  subsumes : (other : const Clause&)  ->  Lit
|
|  Description:
|       Checks if clause subsumes 'other', and at the same time, if it can be
used to simplify 'other' |       by subsumption resolution.
|
|    Result:
|       lit_Error  - No subsumption or simplification
|       lit_Undef  - Clause subsumes 'other'
|       p          - The literal p can be deleted from 'other'
|________________________________________________________________________________________________@*/
inline Lit Clause::subsumes(const Clause& other) const {
    // if (other.size() < size() || (extra.abst & ~other.extra.abst) != 0)
    // if (other.size() < size() || (!learnt() && !other.learnt() && (extra.abst
    // & ~other.extra.abst) != 0))
    assert(!header.learnt);
    assert(!other.header.learnt);
    assert(header.has_extra);
    assert(other.header.has_extra);
    if (other.header.size < header.size ||
        (data[header.size].abs & ~other.data[other.header.size].abs) != 0)
        return lit_Error;

    Lit ret = lit_Undef;
    const Lit* c = (const Lit*)(*this);
    const Lit* d = (const Lit*)other;

    for (unsigned i = 0; i < header.size; i++) {
        // search for c[i] or ~c[i]
        for (unsigned j = 0; j < other.header.size; j++)
            if (c[i] == d[j])
                goto ok;
            else if (ret == lit_Undef && c[i] == ~d[j]) {
                ret = c[i];
                goto ok;
            }

        // did not find it
        return lit_Error;
    ok:;
    }

    return ret;
}

inline void Clause::strengthen(Lit p) {
    remove(*this, p);
    calcAbstraction();
}

//=================================================================================================
}  // namespace Minisat

#endif
