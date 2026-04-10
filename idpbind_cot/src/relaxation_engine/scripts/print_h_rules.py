#!/usr/bin/env python3
import math
from bowerbird2.backend.relaxation_engine.hydrogen_bond_mods import hydrogen_constants as hc


def _build_constant_map(module):
    consts = {}
    for name in dir(module):
        if not name.isupper():
            continue
        # Skip large mapping objects
        if name == 'H_RULES':
            continue
        try:
            val = getattr(module, name)
        except Exception:
            continue
        if isinstance(val, (int, float)):
            consts[name] = float(val)
    return consts


def _repr_value(val, const_map, tol=1e-6):
    # Strings: print quoted
    if isinstance(val, str):
        return f"'{val}'"
    # Numeric: try to match constant name
    if isinstance(val, (int, float)):
        fv = float(val)
        for name, cval in const_map.items():
            if math.isclose(fv, cval, rel_tol=tol, abs_tol=tol):
                return name
        # fallback numeric formatting
        return f"{fv:.4f}"
    # Fallback to repr
    return repr(val)


def main():
    H_RULES = hc.H_RULES
    const_map = _build_constant_map(hc)

    print("H_RULES = {")
    for res, rules in H_RULES.items():
        print(f"    '{res}': {{")
        for h_name, rule in rules.items():
            gg, g, p, r, theta, chi = rule
            r_repr = _repr_value(r, const_map)
            t_repr = _repr_value(theta, const_map)
            c_repr = _repr_value(chi, const_map)
            print(f"        '{h_name}': ('{gg}', '{g}', '{p}', {r_repr}, {t_repr}, {c_repr}),")
        print("    },")
    print("}")


if __name__ == '__main__':
    main()
