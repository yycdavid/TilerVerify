/***************************************************************************************[Solver.cc]
Copyright (c) 2003-2006, Niklas Een, Niklas Sorensson
Copyright (c) 2007-2010, Niklas Sorensson
Copyright (c) 2020-2020, Kai Jia

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

#pragma once

#include <unordered_map>
#include "minisat/core/SolverTypes.h"

namespace Minisat {

//! record and rebuild clauses to cache part of a problem
class ClauseRecorder {
    struct IneqAssignClause {
        vec<Lit> lits;
        int bound;
        Lit dst;
    };

    int m_nr_var = 0;
    std::vector<vec<Lit>> m_disj_clause;
    std::vector<IneqAssignClause> m_leq_assign_clause, m_geq_assign_clause;
    mutable vec<Lit> m_add_tmp;

    void update_nr_var(Lit l) {
        int v = var(l) + 1;
        if (v > m_nr_var) {
            m_nr_var = v;
        }
    }

    void add_ineq_assign(IneqAssignClause& target, const vec<Lit>& lits,
                         int bound, Lit dst) {
        lits.copyTo(target.lits);
        target.bound = bound;
        target.dst = dst;

        update_nr_var(dst);
        for (auto i : lits) {
            update_nr_var(i);
        }
    }

    vec<Lit>& mutable_lit(const vec<Lit>& l) const {
        l.copyTo(m_add_tmp);
        return m_add_tmp;
    }

    std::unordered_map<Var, int> m_var_preference;

public:
    void add_disjuction(const vec<Lit>& lits) {
        m_disj_clause.emplace_back();
        lits.copyTo(m_disj_clause.back());

        for (auto&& i : lits) {
            update_nr_var(i);
        }
    }

    void add_leq_assign(const vec<Lit>& lits, int bound, Lit dst) {
        m_leq_assign_clause.emplace_back();
        add_ineq_assign(m_leq_assign_clause.back(), lits, bound, dst);
    }

    void add_geq_assign(const vec<Lit>& lits, int bound, Lit dst) {
        m_geq_assign_clause.emplace_back();
        add_ineq_assign(m_geq_assign_clause.back(), lits, bound, dst);
    }

    template <class Solver>
    void replay(Solver& solver) const {
        for (int i = 0; i < m_nr_var; ++i) {
            solver.newVar();
        }

        for (auto&& i : m_disj_clause) {
            solver.addClause_(mutable_lit(i));
        }
        for (auto&& i : m_leq_assign_clause) {
            solver.addLeqAssign_(mutable_lit(i.lits), i.bound, i.dst);
        }
        for (auto&& i : m_geq_assign_clause) {
            solver.addGeqAssign_(mutable_lit(i.lits), i.bound, i.dst);
        }

        for (auto i : m_var_preference) {
            solver.setVarPreference(i.first, i.second);
        }
    }

    void add_var_preference(Var var, int pref) { m_var_preference[var] = pref; }

    //! number of recorded vars
    int nr_var() const { return m_nr_var; }
};
}  // namespace Minisat
