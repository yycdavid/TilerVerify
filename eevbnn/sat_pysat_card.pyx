# -*- coding: utf-8 -*-
# cython: language_level=3

"""utilities for encoding cardinality constraints"""

cdef class CardEncoder:
    """cardinality constraint encoder"""

    cdef object _add_clause
    cdef object _add_leq_assign
    cdef object _add_geq_assign
    cdef object _env

    def __init__(self, env):
        self._env = env
        solver = env._solver.solver
        if env._all_clauses is None:
            self._add_clause = solver.add_clause
        else:
            def add_clause(c, *, _add=solver.add_clause,
                           _rec=env._add_clause_record):
                _add(c)
                _rec(c.copy())
            self._add_clause = add_clause

        if hasattr(solver, 'add_leq_assign'):
            if env._all_clauses is None:
                self._add_leq_assign = solver.add_leq_assign
                self._add_geq_assign = solver.add_geq_assign
            else:
                def add_leq_assign(lits, bound, dst, *,
                                   add=solver.add_leq_assign,
                                   rec=env._add_clause_record):
                    add(lits, bound, dst)
                    rec(lits + ['<=', bound, '#', dst])
                def add_geq_assign(lits, bound, dst, *,
                                   add=solver.add_geq_assign,
                                   rec=env._add_clause_record):
                    add(lits, bound, dst)
                    rec(lits + ['>=', bound, '#', dst])
                self._add_leq_assign = add_leq_assign
                self._add_geq_assign = add_geq_assign
        else:
            self._add_leq_assign = None
            self._add_geq_assign = None

    cdef _add_var_defeq(self, int var, list clauses):
        cdef int a0, a1, b0, b1
        cdef list i

        for i in clauses:
            i.append(-var)
            self._add_clause(i)
            i.pop()

        if len(clauses) == 1:
            (a0, a1), = clauses
            neg = [[-a0], [-a1]]
        elif len(clauses) == 2:
            if len(clauses[0]) == 1:
                (a0, ), (b0, ) = clauses
                neg = [[-a0, -b0]]
            else:
                (a0, a1), (b0, b1) = clauses
                a0, a1, b0, b1 = -a0, -a1, -b0, -b1
                neg = [[a0, b0], [a0, b1], [a1, b0], [a1, b1]]

        for i in neg:
            i.append(var)
            self._add_clause(i)

    cdef _make_true(self, name):
        ret = self._env.new_var(name).var
        self._add_clause([ret])
        return ret

    cdef _make_false(self, name):
        ret = self._env.new_var(name).var
        self._add_clause([-ret])
        return ret

    def make_geq(self, list lits, int bound, name) -> int:
        cdef int i
        cdef int b
        cdef int varcnt

        if self._add_geq_assign is not None:
            ret = self._env.new_var(name).var
            self._add_geq_assign(lits, bound, ret)
            return ret

        if bound <= 0:
            return self._make_true(name)
        if bound > len(lits):
            return self._make_false(name)

        if bound > len(lits) // 2:
            return self.make_leq([-i for i in lits], len(lits) - bound, name)

        varcnt = self._env._varcnt
        aux = [[None, lits[0]]]
        for i in range(1, len(lits)):
            cur = [None]
            for b in range(1, min(bound, i + 1) + 1):
                varcnt += 1
                cur.append(varcnt)
                if b == i + 1:
                    cls = [
                        [aux[i - 1][b - 1]],
                        [lits[i]]
                    ]
                elif b > 1:
                    cls = [
                        [aux[i - 1][b], aux[i - 1][b - 1]],
                        [aux[i - 1][b], lits[i]]
                    ]
                else:
                    cls = [
                        [aux[i - 1][b], lits[i]]
                    ]

                self._add_var_defeq(varcnt, cls)
            aux.append(cur)

        self._env._varcnt = varcnt
        ret = aux[len(lits) - 1][bound]
        return ret

    def make_leq(self, list lits, int bound, name) -> int:
        cdef int i
        cdef int b
        cdef int varcnt

        if self._add_leq_assign is not None:
            ret = self._env.new_var(name).var
            self._add_leq_assign(lits, bound, ret)
            return ret

        if bound < 0:
            return self._make_false(name)
        if bound >= len(lits):
            return self._make_true(name)

        if bound > len(lits) // 2:
            return self.make_geq([-i for i in lits], len(lits) - bound, name)

        varcnt = self._env._varcnt
        aux = [[-lits[0]]]
        for i in range(1, len(lits)):
            cur = []
            for b in range(min(i, bound) + 1):
                varcnt += 1
                cur.append(varcnt)
                if b == i:
                    cls = [
                        [aux[i - 1][b - 1], -lits[i]]
                    ]
                elif b >= 1:
                    cls = [
                        [aux[i - 1][b - 1], aux[i - 1][b]],
                        [aux[i - 1][b - 1], -lits[i]]
                    ]
                else:
                    cls = [
                        [aux[i - 1][b]],
                        [-lits[i]]
                    ]

                self._add_var_defeq(varcnt, cls)
            aux.append(cur)

        self._env._varcnt = varcnt
        ret = aux[len(lits) - 1][bound]
        return ret
