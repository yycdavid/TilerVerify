/****************************************************************************************[Dimacs.h]
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

#ifndef Minisat_Dimacs_h
#define Minisat_Dimacs_h

#include <stdio.h>

#include "minisat/core/SolverTypes.h"
#include "minisat/utils/ParseUtils.h"

namespace Minisat {

//=================================================================================================
// DIMACS Parser:
// (jiakai) Extended format:
//      1. Assignment of inequality: lit0 lit1 ... litn (<=|>=) bound # lit_dst
//      2. Var preference: c vpref lit0 pref0 ... litn prefn 0

template <class B, class Solver>
static bool readClause(B& in, Solver& S, vec<Lit>& lits) {
    lits.clear();

    auto get_lit = [&S](int v) -> Lit {
        int var = abs(v) - 1;
        while (var >= S.nVars())
            S.newVar();
        return (v > 0) ? mkLit(var) : ~mkLit(var);
    };

    for (;;) {
        skipWhitespace(in);
        if (*in == '>' || *in == '<') {
            // parse inequalities
            char op = *in;
            ++in;
            if (*in != '=') {
                fprintf(stderr,
                        "PARSE ERROR! Unexpected char in inequality: %c\n",
                        *in),
                        exit(3);
            }
            ++in;
            int bound = parseInt(in);
            skipWhitespace(in);
            if (*in != '#') {
                fprintf(stderr,
                        "PARSE ERROR! Unexpected char in inequality assign: "
                        "%c\n",
                        *in),
                        exit(3);
            }
            ++in;
            Lit dst = get_lit(parseInt(in));
            if (op == '<') {
                S.addLeqAssign_(lits, bound, dst);
            } else {
                S.addGeqAssign_(lits, bound, dst);
            }
            return false;
        }

        int parsed_lit = parseInt(in);
        if (parsed_lit == 0)
            break;
        lits.push(get_lit(parsed_lit));
    }
    return true;
}

template <class B, class Solver>
static void parse_DIMACS_main(B& in, Solver& S) {
    vec<Lit> lits;
    int vars = 0;
    int clauses = 0;
    int cnt = 0;
    size_t cmdlen;
    char cmd[6];
    for (;;) {
        skipWhitespace(in);
        switch (*in) {
            case EOF:
                if (vars != S.nVars()) {
                    fprintf(stderr,
                            "WARNING! DIMACS header mismatch: wrong number of "
                            "variables.\n");
                }
                if (cnt != clauses) {
                    fprintf(stderr,
                            "WARNING! DIMACS header mismatch: wrong number of "
                            "clauses.\n");
                }
                return;
            case 'p':
                if (eagerMatch(in, "p cnf")) {
                    vars = parseInt(in);
                    clauses = parseInt(in);
                    // SATRACE'06 hack
                    // if (clauses > 4000000)
                    //     S.eliminate(true);
                } else {
                    fprintf(stderr, "PARSE ERROR! Unexpected char: %c\n", *in);
                    exit(3);
                }
                break;
            case 'c':
                ++in;
                skipWhitespace(in);
                cmdlen = 0;
                while (*in != ' ' && *in != '\n' && cmdlen < sizeof(cmd) - 1) {
                    cmd[cmdlen++] = *in;
                    ++in;
                }
                cmd[cmdlen] = 0;
                if (!strcmp(cmd, "vpref")) {
                    for (;;) {
                        int lit = parseInt(in);
                        if (!lit) {
                            break;
                        }
                        int pref = parseInt(in);
                        S.setVarPreference(std::abs(lit) - 1, pref);
                    }
                } else if (!strcmp(cmd, "vname")) {
                    int lit = parseInt(in);
                    std::string name;
                    skipWhitespace(in);
                    while (*in != '\n') {
                        name.append(1, *in);
                        ++in;
                    }
                    ++in;
                    S.var_names[std::abs(lit) - 1] = std::move(name);
                } else {
                    skipLine(in);
                }
                break;
            default:
                cnt++;
                if (readClause(in, S, lits)) {
                    S.addClause_(lits);
                }
        }
    }
}

// Inserts problem into solver.
//
template <class Solver>
static void parse_DIMACS(gzFile input_stream, Solver& S) {
    StreamBuffer in(input_stream);
    parse_DIMACS_main(in, S);
}

//=================================================================================================
}  // namespace Minisat

#endif
