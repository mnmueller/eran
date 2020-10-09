from elina_scalar import *
from elina_dimension import *
from elina_linexpr0 import *
from elina_abstract0 import *
from fppoly import *
from fconv import *

import numpy as np
import time
import itertools
import multiprocessing
import math

from config import config

"""
For representing the constraints CDD format is used
http://web.mit.edu/sage/export/cddlib-094b.dfsg/doc/cddlibman.ps:
each row represents b + Ax >= 0
example: 2*x_1 - 3*x_2 >= 1 translates to [-1, 2, -3]
"""


def generate_linexpr0(offset, varids, coeffs):
    # returns ELINA expression, equivalent to sum_i(varids[i]*coeffs[i])
    assert len(varids) == len(coeffs)
    n = len(varids)

    linexpr0 = elina_linexpr0_alloc(ElinaLinexprDiscr.ELINA_LINEXPR_SPARSE, n)
    cst = pointer(linexpr0.contents.cst)
    elina_scalar_set_double(cst.contents.val.scalar, 0)

    for i, (x, coeffx) in enumerate(zip(varids, coeffs)):
        linterm = pointer(linexpr0.contents.p.linterm[i])
        linterm.contents.dim = ElinaDim(offset + x)
        coeff = pointer(linterm.contents.coeff)
        elina_scalar_set_double(coeff.contents.val.scalar, coeffx)

    return linexpr0


class KAct:
    def __init__(self, input_hrep):
        assert KAct.type in ["ReLU", "Tanh", "Sigmoid"]
        self.k = len(input_hrep[0]) - 1
        self.input_hrep = np.array(input_hrep)

        if KAct.type == "ReLU":
            self.cons = fkrelu(self.input_hrep)
        elif KAct.type == "Tanh":
            self.cons = fktanh(self.input_hrep)
        else:
            self.cons = fksigm(self.input_hrep)


def make_kactivation_obj(input_hrep):
    return KAct(input_hrep)


def get_ineqs_zono(varsid):
    input_hrep = []

    # Get bounds on linear expressions over variables before relu
    # Order of coefficients determined by logic here
    for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
        if all(c==0 for c in coeffs):
            continue

        linexpr0 = generate_linexpr0(KAct.offset, varsid, coeffs)
        element = elina_abstract0_assign_linexpr_array(KAct.man, True, Krelu.element,
                                                       Krelu.tdim, linexpr0, 1, None)
        bound_linexpr = elina_abstract0_bound_dimension(KAct.man, KAct.element,
                                                        KAct.offset + KAct.length)
        upper_bound = bound_linexpr.contents.sup.contents.val.dbl
        input_hrep.append([upper_bound] + [-c for c in coeffs])
    return input_hrep


def sparse_heuristic_with_cutoff(all_vars, areas):
    assert len(all_vars) == len(areas)
    K = 3
    sparse_n = config.sparse_n
    cutoff = 0.05
    print("sparse n", sparse_n)
    # Sort vars by descending area
    all_vars = sorted(all_vars, key=lambda var: -areas[var])

    vars_above_cutoff = [i for i in all_vars if areas[i] >= cutoff]

    kact_args = []
    while len(vars_above_cutoff) > 0 and config.sparse_n >= K:
        grouplen = min(sparse_n, len(vars_above_cutoff))
        group = vars_above_cutoff[:grouplen]
        vars_above_cutoff = vars_above_cutoff[grouplen:]
        if grouplen <= K:
            kact_args.append(group)
        else:
            sparsed_combs = generate_sparse_cover(grouplen, K)
            for comb in sparsed_combs:
                kact_args.append(tuple([group[i] for i in comb]))

    # Also just apply 1-relu for every var.
    for var in all_vars:
        kact_args.append([var])

    return kact_args


def encode_kactivation_cons(nn, man, element, offset, layerno, length, lbi, ubi, constraint_groups, need_pop, domain, activation_type):
    import deepzono_nodes as dn
    if need_pop:
        constraint_groups.pop()

    lbi = np.asarray(lbi, dtype=np.double)
    ubi = np.asarray(ubi, dtype=np.double)

    candidate_vars = [i for i in range(length) if lbi[i] < 0 < ubi[i]]
    candidate_vars_areas = {var: -lbi[var] * ubi[var] for var in candidate_vars}
    # Sort vars by descending area
    candidate_vars = sorted(candidate_vars, key=lambda var: -candidate_vars_areas[var])

    # Use sparse heuristic to select args
    kact_args = sparse_heuristic_with_cutoff(candidate_vars, candidate_vars_areas)

    kact_cons = []
    tdim = ElinaDim(offset+length)
    if domain == 'refinezono':
        element = dn.add_dimensions(man,element,offset+length,1)

    KAct.man = man
    KAct.element = element
    KAct.tdim = tdim
    KAct.length = length
    KAct.layerno = layerno
    KAct.offset = offset
    KAct.domain = domain
    KAct.type = activation_type

    start = time.time()

    if domain == 'refinezono':
        with multiprocessing.Pool(config.numproc) as pool:
            input_hrep_array = pool.map(get_ineqs_zono, kact_args)
    else:
        total_size = 0
        for varsid in kact_args:
            size = 3**len(varsid) - 1
            total_size = total_size + size

        linexpr0 = elina_linexpr0_array_alloc(total_size)
        i = 0
        for varsid in kact_args:
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue

                linexpr0[i] = generate_linexpr0(offset, varsid, coeffs)
                i = i + 1
        upper_bound = get_upper_bound_for_linexpr0(man,element,linexpr0, total_size, layerno)
        i=0
        input_hrep_array = []
        for varsid in kact_args:
            input_hrep = []
            for coeffs in itertools.product([-1, 0, 1], repeat=len(varsid)):
                if all(c == 0 for c in coeffs):
                    continue
                input_hrep.append([upper_bound[i]] + [-c for c in coeffs])
                i = i + 1
            input_hrep_array.append(input_hrep)

    with multiprocessing.Pool(config.numproc) as pool:
        kact_results = pool.map(make_kactivation_obj, input_hrep_array)

    gid = 0
    for inst in kact_results:
        varsid = kact_args[gid]
        inst.varsid = varsid
        kact_cons.append(inst)
        gid = gid+1
    end = time.time()

    if config.debug:
        print('kactivation time spent: ' + str(end-start))
    if domain == 'refinezono':
        element = dn.remove_dimensions(man, element, offset+length, 1)

    constraint_groups.append(kact_cons)
