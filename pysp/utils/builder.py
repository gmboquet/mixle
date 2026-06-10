from pysp.stats import *
from typing import Optional, Tuple

def read_index_csv(filename: str):
    fin = open(filename, 'r')
    lines = map(lambda v: v.split('#', 1)[0].split(',', 3), fin.read().split('\n'))
    fin.close()
    lines = filter(lambda v: len(v) == 4, lines)

    return lines


def get_indexed_rdd_pne(field_info=None, filename=None):
    if filename is not None and field_info is None:
        field_info = read_index_csv(filename)

    def entry_lambda(idx, map_str):
        if map_str != '':
            temp_lambda0 = eval('lambda x: ' + map_str)
            temp_lambda = lambda u: temp_lambda0(u[idx])
        else:
            temp_lambda = lambda u: u[idx]

        def ff(entry):
            rv = temp_lambda(entry)
            return rv

        return ff

    parser_list = []
    estimator_list = []
    max_idx = -1

    for entry in field_info:
        idx, name, lam, dist = entry
        estimator = eval(dist)

        if estimator is not None:
            idx_i = int(idx)
            parser_list.append(entry_lambda(idx_i, lam.strip()))
            estimator_list.append(estimator)
            max_idx = idx_i if idx_i > max_idx else max_idx

    def line_parser(line: str):
        parts = line.split(',')
        if len(parts) < (max_idx + 1):
            return None
        else:
            return tuple([parser(parts) for parser in parser_list])

    estimator = CompositeEstimator(tuple(estimator_list))

    return estimator, line_parser
