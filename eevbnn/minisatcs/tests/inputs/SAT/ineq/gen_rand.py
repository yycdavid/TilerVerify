#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import argparse
import operator

def main():
    parser = argparse.ArgumentParser(
        description='generate random input with inequalities',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('nvar', type=int)
    parser.add_argument('output', help='output file name')
    parser.add_argument('--min-incl', type=int, default=5,
                        help='minimal number of clauses containing a given var')
    parser.add_argument('--cl-size-min', type=int, default=2,
                        help='minimal number of literals in a clause')
    parser.add_argument('--cl-bias-max', type=int, default=10,
                        help='max clause bias value')
    parser.add_argument('--cl-bias-offset', type=int, default=3,
                        help='max random offset of bias value')
    parser.add_argument('--seed', type=int, default=np.random.randint(2**32),
                        help='rng seed')
    args = parser.parse_args()

    assignment = np.random.randint(2, size=args.nvar)
    incl_cnt = np.zeros(args.nvar, dtype=np.uint32)
    clauses = []

    bias_th0 = args.cl_bias_max
    rng = np.random.RandomState(args.seed)

    while incl_cnt.max() < args.min_incl:
        cur_vars = rng.choice(
            args.nvar, size=rng.randint(args.cl_size_min, args.nvar + 1),
            replace=False)
        incl_cnt[cur_vars] += 1
        mask = assignment[cur_vars]
        cur_val = mask.sum()
        bias_th1 = cur_vars.size - bias_th0
        cur_vars += 1

        # fix cur_val range by negating some vars
        if bias_th0 < cur_val < bias_th1:
            offset = rng.randint(-args.cl_bias_offset,
                                       args.cl_bias_offset + 1)
            if cur_val < (bias_th0 + bias_th1) / 2:
                # change some true lits to false
                nr = max(cur_val - bias_th0 + offset, 0)
                sel, = np.where(mask == 1)
                cur_val -= nr
            else:
                # change some false lits to false
                nr = max(bias_th1 - cur_val + offset, 0)
                sel, = np.where(mask == 0)
                cur_val += nr

            sel = rng.choice(sel, size=nr, replace=False)
            cur_vars[sel] *= -1

        cur_clause = list(cur_vars)
        if rng.randint(2):
            cur_clause.append('<=')
            cmpr = operator.le
        else:
            cur_clause.append('>=')
            cmpr = operator.ge
        offset = rng.randint(-args.cl_bias_offset,
                             args.cl_bias_offset + 1)
        cur_clause.append(cur_val + offset)
        cur_clause.append('#')

        dst = rng.randint(1, args.nvar + 1)
        if int(cmpr(0, offset)) != assignment[dst - 1]:
            dst = -dst
        cur_clause.append(dst)
        clauses.append(' '.join(map(str, cur_clause)))

    with open(args.output, 'w') as fout:
        fout.write(f'c args: {args}\n')
        fout.write('c assignment: ')
        fout.write(' '.join(str(i + 1) if j else str(-(i + 1))
                            for i, j in enumerate(assignment)))
        fout.write('\n')
        fout.write(f'p cnf {args.nvar} {len(clauses)}\n')
        for i in clauses:
            fout.write(i)
            fout.write('\n')

if __name__ == '__main__':
    main()
