/****************************************************************************************[Solver.h]
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

#ifndef Minisat_Solver_h
#define Minisat_Solver_h

#include "minisat/core/SolverTypes.h"
#include "minisat/mtl/Alg.h"
#include "minisat/mtl/Heap.h"
#include "minisat/mtl/Vec.h"
#include "minisat/utils/Random.h"

#include <string>
#include <unordered_map>
#include <utility>

namespace Minisat {

//=================================================================================================
// Solver -- the main class:

class Solver;

//! remove vars and corresponding clauses that are referenced by at most one
//! clause
class DeadVarRemover {
    //! refcnt of a single var
    struct RefCnt {
        // tot refcnt: total number of clauses referring to this var
        int tot;
        //! removable references to this var: disjunctive clause or dst of leq
        //! assign
        int removable;

        bool safe_to_remove() const {
            return !tot || (tot == 1 && removable == 1);
        }
    };
    struct LeqRec {
        vec<Lit> lits;
        int bound;
        Lit dst;
    };

    Solver* const m_solver;
    bool m_enabled = true;
    vec<RefCnt> m_var_refcnt;
    //! vars and corresponding clauses that reference it
    std::vector<std::pair<Var, CRef>> m_var2cref;
    //! queue of vars to be removed
    vec<Var> m_to_remove;
    std::vector<LeqRec> m_leq_to_fix;

    inline void incr_refcnt(CRef cref);
    inline void remove_clause_and_decr_refcnt(Var src_var, CRef cref);

    //! try to find the remaining unremoved cref to a var
    inline std::optional<CRef> find_remaining_cref(Var var) const;

    //! if cnt indicates that var is safe to be removed, then add it to the
    //! queue
    inline void add_to_remove_if_safe(RefCnt& cnt, Var var);

    void clean_removed(vec<CRef>& cs);

public:
    explicit DeadVarRemover(Solver* solver) : m_solver{solver} {}

    void disable() { m_enabled = false; }

    void simplify();

    //! to be called after a solution is found, so assignments of removed vars
    //! can be fixed
    void fix_var_assignments();
};

class Solver {
public:
    // Constructor/Destructor:
    //
    Solver();
    virtual ~Solver();

    // Problem specification:
    //
    Var newVar(bool polarity = true,
               bool dvar = true);  // Add a new variable with parameters
                                   // specifying variable mode.

    bool addClause(const vec<Lit>& ps);  // Add a clause to the solver.
    //! Add the empty clause, making the solver contradictory.
    bool addEmptyClause();
    bool addClause(Lit p);                // Add a unit clause to the solver.
    bool addClause(Lit p, Lit q);         // Add a binary clause to the solver.
    bool addClause(Lit p, Lit q, Lit r);  // Add a ternary clause to the solver.
    bool addClause_(vec<Lit>& ps);  // Add a clause to the solver without making
                                    // superflous internal copy. Will change the
                                    // passed vector 'ps'.
    //! add dst = (src[0]^src_neg) & (src[1]^src_neg) & ...
    template <bool src_neg = false>
    bool addClauseReifiedConjunction(Lit dst, const Lit* src, int size);

    //! Add dst = (sum(ps) <= bound) to the solver
    bool addLeqAssign_(vec<Lit>& ps, int bound, Lit dst);

    //! Add dst = (sum(ps) >= bound) to the solver
    bool addGeqAssign_(vec<Lit>& ps, int bound, Lit dst) {
        return addLeqAssign_(ps, bound - 1, ~dst);
    }

    // Solving:
    //
    // Removes already satisfied clauses.
    bool simplify();
    bool solve(const vec<Lit>& assumps);  // Search for a model that respects a
                                          // given set of assumptions.
    lbool solveLimited(
            const vec<Lit>&
                    assumps);  // Search for a model that respects a given set
                               // of assumptions (With resource constraints).
    bool solve();              // Search without assumptions.
    bool solve(Lit p);  // Search for a model that respects a single assumption.
    bool solve(Lit p,
               Lit q);  // Search for a model that respects two assumptions.
    bool solve(Lit p, Lit q,
               Lit r);  // Search for a model that respects three assumptions.
    bool okay() const;  // FALSE means solver is in a conflicting state

    void toDimacs(
            FILE* f,
            const vec<Lit>& assumps);  // Write CNF to file in DIMACS-format.
    void toDimacs(const char* file, const vec<Lit>& assumps);
    void toDimacs(FILE* f, Clause& c, vec<Var>& map, Var& max);

    // Convenience versions of 'toDimacs()':
    void toDimacs(const char* file);
    void toDimacs(const char* file, Lit p);
    void toDimacs(const char* file, Lit p, Lit q);
    void toDimacs(const char* file, Lit p, Lit q, Lit r);

    // Variable mode:
    //
    //! Declare which polarity the decision heuristic should use for a variable.
    //! Requires mode 'polarity_user'
    void setPolarity(Var v, bool b);
    //! set var preference to break ties with equal activity; less value means
    //! preferred
    void setVarPreference(Var v, int p);
    //! Declare if a variable should be eligible for selection in the decision
    //! heuristic.
    void setDecisionVar(Var v, bool b);

    // Read state:
    //
    lbool value(Var x) const;  // The current value of a variable.
    lbool value(Lit p) const;  // The current value of a literal.
    lbool modelValue(
            Var x) const;  // The value of a variable in the last model. The
                           // last call to solve must have been satisfiable.
    lbool modelValue(
            Lit p) const;  // The value of a literal in the last model. The last
                           // call to solve must have been satisfiable.
    int nAssigns() const;  // The current number of assigned literals.
    int nClauses() const;  // The current number of original clauses.
    int nLeqClauses() const;  // The current number of original LEQ clauses.
    int nLearnts() const;     // The current number of learnt clauses.
    int nVars() const;        // The current number of variables.
    int nFreeVars() const;

    // Resource contraints:
    //
    void setConfBudget(int64_t x);
    void setPropBudget(int64_t x);
    void budgetOff();
    void interrupt();  // Trigger a (potentially asynchronous) interruption of
                       // the solver.
    void clearInterrupt();  // Clear interrupt indicator flag.

    // Memory managment:
    //
    virtual void garbageCollect();
    void checkGarbage(double gf);
    void checkGarbage();

    // Extra results: (read-only member variable)
    //
    vec<lbool> model;   // If problem is satisfiable, this vector contains the
                        // model (if any).
    vec<Lit> conflict;  // If problem is unsatisfiable (possibly under
                        // assumptions), this vector represent the final
                        // conflict clause expressed in the assumptions.

    // Mode of operation:
    //
    int verbosity;
    double var_decay;
    double clause_decay;
    double random_var_freq;
    bool luby_restart;
    int ccmin_mode;  // Controls conflict clause minimization (0=none, 1=basic,
                     // 2=deep).
    int phase_saving;  // Controls the level of phase saving (0=none, 1=limited,
                       // 2=full).
    bool rnd_pol;      // Use random polarities for branching heuristics.
    bool rnd_init_act;    // Initialize variable activities with a small random
                          // value.
    double garbage_frac;  // The fraction of wasted memory allowed before a
                          // garbage collection is triggered.

    int restart_first;         // The initial restart limit. (default 100)
    double restart_inc;        // The factor with which the restart limit is
                               // multiplied in each restart. (default 1.5)
    double learntsize_factor;  // The intitial limit for learnt clauses is a
                               // factor of the original clauses. (default 1 /
                               // 3)
    double learntsize_inc;  // The limit for learnt clauses is multiplied with
                            // this factor each restart. (default 1.1)

    int learntsize_adjust_start_confl;
    double learntsize_adjust_inc;

    //! this can be set for better debug output
    std::unordered_map<int, std::string> var_names;

    // Statistics: (read-only member variable)
    //
    uint64_t solves, starts, decisions, rnd_decisions, propagations, conflicts;
    uint64_t dec_vars, clauses_literals, learnts_literals, max_literals,
            tot_literals;

protected:
    using abstract_level_set_t = uint_fast32_t;

    // Helper structures:
    //
    struct VarData {
        CRef reason;
        int level;
    };

    struct Watcher {
        CRef cref;
        Lit blocker;
        Watcher(CRef cr, Lit p) : cref(cr), blocker(p) {}
        bool operator==(const Watcher& w) const { return cref == w.cref; }
        bool operator!=(const Watcher& w) const { return cref != w.cref; }
    };

    //! watcher for LEQ clauses
    struct LeqWatcher;

    //! modification log of LeqStatus
    struct LeqStatusModLog;

    //! used in trail_lim
    struct TrailSep {
        int lit, leq;
    };

    struct WatcherRefreshDisj {
        const ClauseAllocator& ca;
        WatcherRefreshDisj(const ClauseAllocator& _ca) : ca(_ca) {}

        bool operator()(const Watcher& w) const {
            return ca[w.cref].mark() == 1;
        }
    };
    struct WatcherRefreshLeq {
        const ClauseAllocator& ca;
        WatcherRefreshLeq(const ClauseAllocator& _ca) : ca(_ca) {}

        inline bool operator()(LeqWatcher& w) const;
    };

    class VarOrderLt {
        static constexpr double eps = 1e-6;
        const Solver* m_solver;

    public:
        inline bool operator()(Var x, Var y) const;
        explicit VarOrderLt(const Solver* solver) : m_solver{solver} {}
    };

    // Solver state:
    //
    bool ok;  // If FALSE, the constraints are already unsatisfiable. No
              // part of the solver state may be used!
    vec<CRef> clauses;  // List of problem clauses.
    vec<CRef> learnts;  // List of learnt clauses.
    double cla_inc;     // Amount to bump next clause with.
    //! A heuristic measurement of the activity of a variable.
    vec<double> activity;
    //! user-defiend branching order of the vars
    vec<int> var_preference;
    double var_inc;  // Amount to bump next variable with.
    //! 'watches[lit]' is a list of constraints watching 'lit' (will go there if
    //! literal becomes true).
    OccLists<Lit, vec<Watcher>, WatcherRefreshDisj> watches;
    //! constraints watching a var, triggered when it is decided
    OccLists<Var, vec<LeqWatcher>, WatcherRefreshLeq> leq_watches;
    vec<lbool> assigns;  // The current assignments.
    vec<char> polarity;  // The preferred polarity of each variable.
    vec<char> decision;  // Declares if a variable is eligible for selection in
                         // the decision heuristic.
    //! Assignment stack; stores all assigments made in the order they were made
    vec<Lit> trail;
    //! Record of modification on LeqStatus that needs to be undone
    vec<LeqStatusModLog> trail_leq_stat;
    //! Separator indices for different decision levels in 'trail
    vec<TrailSep> trail_lim;
    vec<VarData> vardata;  // Stores reason and level for each variable.
    int qhead;  // Head of queue (as index into the trail -- no more explicit
                // propagation queue in MiniSat).
    int simpDB_assigns;  // Number of top-level assignments since last execution
                         // of 'simplify()'.
    int64_t simpDB_props;  // Remaining number of propagations that must be made
                           // before next execution of 'simplify()'.
    vec<Lit> assumptions;  // Current set of assumptions provided to solve by
                           // the user.
    Heap<VarOrderLt> order_heap;  // A priority queue of variables ordered with
                                  // respect to the variable activity.
    double progress_estimate;     // Set by 'search()'.
    bool remove_satisfied;  // Indicates whether possibly inefficient linear
                            // scan for satisfied clauses should be performed in
                            // 'simplify'.
    //! ensure that global scan would not occur too frequently by requring a
    //! minimum number of propagations
    uint64_t next_remove_satisfied_nr_prop = 0;

    ClauseAllocator ca;

    // Temporaries (to reduce allocation overhead). Each variable is prefixed by
    // the method in which it is used, exept 'seen' wich is used in several
    // places.
    //
    vec<char> seen;
    vec<Lit> analyze_stack;
    vec<Lit> analyze_toclear;
    vec<Lit> add_tmp;

    double max_learnts;
    double learntsize_adjust_confl;
    int learntsize_adjust_cnt;

    // Resource contraints:
    //
    int64_t conflict_budget;     // -1 means no budget.
    int64_t propagation_budget;  // -1 means no budget.
    volatile bool asynch_interrupt;

    // Encapsulated objects
    RandomState random_state;
    friend class DeadVarRemover;
    DeadVarRemover dead_var_remover{this};

    // Main internal methods:
    //
    void insertVarOrder(
            Var x);  // Insert a variable in the decision order priority queue.
    Lit pickBranchLit();      // Return the next decision variable.
    void newDecisionLevel();  // Begins a new decision level.
    //! Enqueue a literal. Assumes value of literal is undefined.
    void uncheckedEnqueue(Lit p, CRef from = CRef_Undef);
    //! revert newly added literals in the queue until size is no larger than
    //! target_size
    void dequeueUntil(int target_size);
    CRef propagate();  // Perform unit propagation. Returns possibly conflicting
                       // clause.
    //! handle LEQ clauses related to the new fact, and return conflict
    CRef propagate_leq(Lit new_fact);
    //! Backtrack until a certain leve, by keeping all assignment at 'level' but
    //! not beyond
    void cancelUntil(int level);
    void analyze(CRef confl, vec<Lit>& out_learnt,
                 int& out_btlevel);  // (bt = backtrack)
    void analyzeFinal(
            Lit p, vec<Lit>& out_conflict);  // COULD THIS BE IMPLEMENTED BY THE
                                             // ORDINARIY "analyze" BY SOME
                                             // REASONABLE GENERALIZATION?
    //! check if a lit is redundant given current visited lits in analyze()
    bool litRedundant(Lit p, abstract_level_set_t abstract_levels);
    lbool search(int nof_conflicts);  // Search for a given number of conflicts.
    lbool solve_();   // Main solve method (assumptions given in 'assumptions').
    void reduceDB();  // Reduce the set of learnt clauses.
    void removeSatisfied(vec<CRef>& cs);  // Shrink 'cs' to contain only
                                          // non-satisfied clauses.
    void rebuildOrderHeap();

    // Maintaining Variable/Clause activity:
    //
    void
    varDecayActivity();  // Decay all variables with the specified factor.
                         // Implemented by increasing the 'bump' value instead.
    void varBumpActivity(
            Var v,
            double inc);  // Increase a variable with the current 'bump' value.
    void varBumpActivity(
            Var v);  // Increase a variable with the current 'bump' value.
    void
    claDecayActivity();  // Decay all clauses with the specified factor.
                         // Implemented by increasing the 'bump' value instead.
    void claBumpActivity(
            Clause& c);  // Increase a clause with the current 'bump' value.

    // Operations on clauses:

    // LEQ clauses:
    //! remove duplicatations in ps and modify ps and bound inplace
    void canonize_leq_clause(vec<Lit>& ps, int& bound);
    //! try constant propagation on LEQ clauses
    std::optional<bool> try_leq_clause_const_prop(const vec<Lit>& ps, Lit dst,
                                                  int bound);
    //! add a new LEQ clause and setup watchers
    void add_leq_and_setup_watchers(vec<Lit>& ps, Lit dst, int bound);

    // disjunction clauses:
    void attachClause(CRef cr);  // Attach a clause to watcher lists.
    void detachClause(
            CRef cr, bool strict = false);  // Detach a clause to watcher lists.
    void removeClause(CRef cr);             // Detach and free a clause.

    //! Check if a clause is a reason for some implication in the current state.
    //! This function should only be called on disjunction clauses
    bool locked_disj(const Clause& c) const;
    //! return whether the clause is satisified given current assignments
    bool satisfied(const Clause& c) const;
    //! try to simplify an LEQ clause; return whether the original clause should
    //! be removed
    bool try_leq_simplify(Clause& c);

    void relocAll(ClauseAllocator& to);

    // Misc:
    //
    int decisionLevel() const;  // Gives the current decisionlevel.
    //! get an abstract representation of the level of var
    abstract_level_set_t abstractLevel(Var x) const;
    // sets of decision levels.
    CRef reason(Var x) const;
    int level(Var x) const;
    double progressEstimate() const;  // DELETE THIS ?? IT'S NOT VERY USEFUL ...
    bool withinBudget() const;

    // Misc helpers:
    //

    //! Move lits with known values matching target value to the beginning
    template <bool sel_true>
    void select_known_lits(Clause& c, int num);

    //! Move lits with known values matching target value to the beginning, and
    //! enqueue unknown lits to be the opposite;
    //! If the number of known vars is greater than nr_known, revert all
    //! enqueued values and return false.
    template <bool sel_true>
    bool select_known_and_imply_unknown(CRef cr, Clause& c, int nr_known);

    //! get use-set var name for debug
    const char* var_name(Var var) const {
        auto iter = var_names.find(var);
        return iter == var_names.end() ? "" : iter->second.c_str();
    }
};

//=================================================================================================
// Implementation of inline methods:

inline CRef Solver::reason(Var x) const {
    return vardata[x].reason;
}
inline int Solver::level(Var x) const {
    return vardata[x].level;
}

inline void Solver::insertVarOrder(Var x) {
    if (!order_heap.inHeap(x) && decision[x])
        order_heap.insert(x);
}

inline void Solver::varDecayActivity() {
    var_inc *= (1 / var_decay);
}
inline void Solver::varBumpActivity(Var v) {
    varBumpActivity(v, var_inc);
}
inline void Solver::varBumpActivity(Var v, double inc) {
    if ((activity[v] += inc) > 1e100) {
        // Rescale:
        for (int i = 0; i < nVars(); i++)
            activity[i] *= 1e-100;
        var_inc *= 1e-100;

        // partial order may have changed, so we simply rebuild the whole heap
        rebuildOrderHeap();
        return;
    }

    // Update order_heap with respect to new activity:
    if (order_heap.inHeap(v))
        order_heap.decrease(v);
}

inline void Solver::claDecayActivity() {
    cla_inc *= (1 / clause_decay);
}
inline void Solver::claBumpActivity(Clause& c) {
    if ((c.activity() += cla_inc) > 1e20) {
        // Rescale:
        for (int i = 0; i < learnts.size(); i++)
            ca[learnts[i]].activity() *= 1e-20;
        cla_inc *= 1e-20;
    }
}

inline void Solver::checkGarbage(void) {
    return checkGarbage(garbage_frac);
}
inline void Solver::checkGarbage(double gf) {
    if (ca.wasted() > ca.size() * gf)
        garbageCollect();
}

inline bool Solver::addClause(const vec<Lit>& ps) {
    ps.copyTo(add_tmp);
    return addClause_(add_tmp);
}
inline bool Solver::addEmptyClause() {
    add_tmp.clear();
    return addClause_(add_tmp);
}
inline bool Solver::addClause(Lit p) {
    add_tmp.clear();
    add_tmp.push(p);
    return addClause_(add_tmp);
}
inline bool Solver::addClause(Lit p, Lit q) {
    add_tmp.clear();
    add_tmp.push(p);
    add_tmp.push(q);
    return addClause_(add_tmp);
}
inline bool Solver::addClause(Lit p, Lit q, Lit r) {
    add_tmp.clear();
    add_tmp.push(p);
    add_tmp.push(q);
    add_tmp.push(r);
    return addClause_(add_tmp);
}
inline bool Solver::locked_disj(const Clause& c) const {
    return value(c[0]) == l_True && reason(var(c[0])) != CRef_Undef &&
           ca.lea(reason(var(c[0]))) == &c;
}
inline void Solver::newDecisionLevel() {
    trail_lim.push({trail.size(), trail_leq_stat.size()});
}

inline int Solver::decisionLevel() const {
    return trail_lim.size();
}
inline Solver::abstract_level_set_t Solver::abstractLevel(Var x) const {
    return static_cast<abstract_level_set_t>(1)
           << (level(x) & (sizeof(abstract_level_set_t) * 8 - 1));
}
inline lbool Solver::value(Var x) const {
    return assigns[x];
}
inline lbool Solver::value(Lit p) const {
    return assigns[var(p)] ^ sign(p);
}
inline lbool Solver::modelValue(Var x) const {
    return model[x];
}
inline lbool Solver::modelValue(Lit p) const {
    return model[var(p)] ^ sign(p);
}
inline int Solver::nAssigns() const {
    return trail.size();
}
inline int Solver::nClauses() const {
    return clauses.size();
}
inline int Solver::nLearnts() const {
    return learnts.size();
}
inline int Solver::nVars() const {
    return vardata.size();
}
inline int Solver::nFreeVars() const {
    return (int)dec_vars -
           (trail_lim.size() == 0 ? trail.size() : trail_lim[0].lit);
}
inline void Solver::setConfBudget(int64_t x) {
    conflict_budget = conflicts + x;
}
inline void Solver::setPropBudget(int64_t x) {
    propagation_budget = propagations + x;
}
inline void Solver::interrupt() {
    asynch_interrupt = true;
}
inline void Solver::clearInterrupt() {
    asynch_interrupt = false;
}
inline void Solver::budgetOff() {
    conflict_budget = propagation_budget = -1;
}
inline bool Solver::withinBudget() const {
    return !asynch_interrupt &&
           (conflict_budget < 0 || conflicts < (uint64_t)conflict_budget) &&
           (propagation_budget < 0 ||
            propagations < (uint64_t)propagation_budget);
}

// FIXME: after the introduction of asynchronous interrruptions the
// solve-versions that return a pure bool do not give a safe interface. Either
// interrupts must be possible to turn off here, or all calls to solve must
// return an 'lbool'. I'm not yet sure which I prefer.
inline bool Solver::solve() {
    budgetOff();
    assumptions.clear();
    return solve_() == l_True;
}
inline bool Solver::solve(Lit p) {
    budgetOff();
    assumptions.clear();
    assumptions.push(p);
    return solve_() == l_True;
}
inline bool Solver::solve(Lit p, Lit q) {
    budgetOff();
    assumptions.clear();
    assumptions.push(p);
    assumptions.push(q);
    return solve_() == l_True;
}
inline bool Solver::solve(Lit p, Lit q, Lit r) {
    budgetOff();
    assumptions.clear();
    assumptions.push(p);
    assumptions.push(q);
    assumptions.push(r);
    return solve_() == l_True;
}
inline bool Solver::solve(const vec<Lit>& assumps) {
    budgetOff();
    assumps.copyTo(assumptions);
    return solve_() == l_True;
}
inline lbool Solver::solveLimited(const vec<Lit>& assumps) {
    assumps.copyTo(assumptions);
    return solve_();
}
inline bool Solver::okay() const {
    return ok;
}

inline void Solver::toDimacs(const char* file) {
    vec<Lit> as;
    toDimacs(file, as);
}
inline void Solver::toDimacs(const char* file, Lit p) {
    vec<Lit> as;
    as.push(p);
    toDimacs(file, as);
}
inline void Solver::toDimacs(const char* file, Lit p, Lit q) {
    vec<Lit> as;
    as.push(p);
    as.push(q);
    toDimacs(file, as);
}
inline void Solver::toDimacs(const char* file, Lit p, Lit q, Lit r) {
    vec<Lit> as;
    as.push(p);
    as.push(q);
    as.push(r);
    toDimacs(file, as);
}

bool Solver::VarOrderLt::operator()(Var x, Var y) const {
    auto ax = m_solver->activity[x], ay = m_solver->activity[y];
    if (ax > ay + eps) {
        return true;
    }
    if (ax > ay - eps) {
        int px = m_solver->var_preference[x], py = m_solver->var_preference[y];
        return px < py || (px == py && x > y);
    }
    return false;
}

//=================================================================================================
// Debug etc:

//=================================================================================================
}  // namespace Minisat

#endif
