#include "minisat/core/Recorder.h"
#include "minisat/core/Solver.h"

#include <signal.h>

#include <algorithm>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <thread>
#include <vector>

class MinisatClauseRecorder : public Minisat::ClauseRecorder {
public:
    // concrete solver type
    void replay(Minisat::Solver& solver) { ClauseRecorder::replay(solver); }
};

class WrappedMinisatSolver : public Minisat::Solver {
    MinisatClauseRecorder* m_recorder = nullptr;

    class ScopedSolverAssign {
        WrappedMinisatSolver** dst;

    public:
        explicit ScopedSolverAssign(WrappedMinisatSolver** dst,
                                    WrappedMinisatSolver* src) {
            *dst = src;
        }
        ~ScopedSolverAssign() { *dst = nullptr; }
    };

    class Timer;

    int m_new_clause_max_var = 0;

    void add_vars() {
        int tgt_nvar = m_new_clause_max_var;
        while (nVars() < tgt_nvar) {
            newVar();
        }
        m_new_clause_max_var = 0;
    }

    Minisat::Lit make_lit(int lit) {
        assert(lit != 0);
        int lv = std::abs(lit);
        m_new_clause_max_var = std::max(m_new_clause_max_var, lv);
        return Minisat::mkLit(lv - 1, lit < 0);
    }

public:
    void set_recorder(MinisatClauseRecorder* recorder) {
        m_recorder = recorder;
    }

    void new_clause_prepare() {
        m_new_clause_max_var = 0;
        add_tmp.clear();
    }

    void new_clause_add_lit(int lit) { add_tmp.push(make_lit(lit)); }

    void new_clause_commit() {
        add_vars();
        if (m_recorder) {
            m_recorder->add_disjuction(add_tmp);
        }
        addClause_(add_tmp);
        add_tmp.clear();
    }

    void new_clause_commit_leq(int bound, int dst) {
        auto dstl = make_lit(dst);
        add_vars();
        if (m_recorder) {
            m_recorder->add_leq_assign(add_tmp, bound, dstl);
        }
        addLeqAssign_(add_tmp, bound, dstl);
        add_tmp.clear();
    }

    void new_clause_commit_geq(int bound, int dst) {
        auto dstl = make_lit(dst);
        add_vars();
        if (m_recorder) {
            m_recorder->add_geq_assign(add_tmp, bound, dstl);
        }
        addGeqAssign_(add_tmp, bound, dstl);
        add_tmp.clear();
    }

    void set_var_preference(int x, int p) {
        int var = std::abs(x) - 1;
        while (nVars() <= var) {
            newVar();
        }
        setVarPreference(var, p);
        if (m_recorder) {
            m_recorder->add_var_preference(var, p);
        }
    }

    void set_var_name(int x, const char* name) {
        var_names[std::abs(x) - 1] = name;
    }

    std::vector<int> get_model() const {
        std::vector<int> ret;
        for (int i = 0; i < model.size(); ++i) {
            if (model[i] == Minisat::l_True) {
                ret.push_back(i + 1);
            } else if (model[i] == Minisat::l_False) {
                ret.push_back(-i - 1);
            }
        }
        return ret;
    }

    //! return -1 for timeout, 0 for unsat, 1 for sat. set timeout < 0 to
    //! disable
    inline int solve_with_signal(bool setup, double timeout);
};

class WrappedMinisatSolver::Timer {
    bool m_canceled = true;
    double m_timeout;
    std::function<void()> m_callback;
    std::thread m_worker;
    std::condition_variable m_cv;
    std::mutex m_mtx;

public:
    Timer(double timeout, std::function<void()> cb)
            : m_timeout{timeout}, m_callback{std::move(cb)} {
        if (timeout > 0) {
            m_canceled = false;
            auto work = [this]() {
                std::unique_lock<std::mutex> lk{m_mtx};
                if (m_canceled) {
                    return;
                }
                m_cv.wait_for(lk, std::chrono::duration<double>{m_timeout});
                if (!m_canceled) {
                    m_callback();
                }
            };
            std::thread t{work};
            m_worker.swap(t);
        }
    }
    ~Timer() { cancel(); }
    void cancel() {
        if (m_canceled) {
            return;
        }
        {
            std::lock_guard<std::mutex> lg{m_mtx};
            m_canceled = true;
            m_cv.notify_one();
        }
        m_worker.join();
    }
};

int WrappedMinisatSolver::solve_with_signal(bool setup, double timeout) {
    static WrappedMinisatSolver* g_solver = nullptr;
    static auto on_sig = [](int) {
        if (g_solver) {
            g_solver->interrupt();
        }
    };
    bool is_tle = false;
    Timer timer{timeout, [this, &is_tle]() {
                    is_tle = true;
                    interrupt();
                }};

    clearInterrupt();
    struct sigaction old_action;
    ScopedSolverAssign g_solver_assign{&g_solver, this};
    if (setup) {
        struct sigaction act;
        memset(&act, 0, sizeof(act));
        act.sa_handler = on_sig;
        if (sigaction(SIGINT, &act, &old_action)) {
            char msg[1024];
            snprintf(msg, sizeof(msg), "failed to setup signal handler: %s",
                     strerror(errno));
            throw std::runtime_error{msg};
        }
    }
    budgetOff();
    assumptions.clear();
    auto ret = solve_();
    if (sigaction(SIGINT, &old_action, nullptr)) {
        char msg[1024];
        snprintf(msg, sizeof(msg), "failed to restore signal handler: %s",
                 strerror(errno));
        throw std::runtime_error{msg};
    }

    if (is_tle) {
        return -1;
    }

    if (ret.is_not_undef()) {
        return ret.as_bool();
    }
    throw std::runtime_error("computation interrupted");
}

