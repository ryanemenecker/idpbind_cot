import math

def test_obc(r, rho_i, sj):
    # piecewise
    if r > rho_i + sj: # regime 1
        return 0.5 * (sj / (r**2 - sj**2) + (1.0/(2*r)) * math.log((r - sj)/(r + sj)))
    elif r > abs(rho_i - sj) and r < rho_i + sj: # regime 2
        return 0.25 * (1/rho_i - 1/(r + sj) - (r**2 - sj**2 + rho_i**2)/(2*r*rho_i**2) + (1.0/r) * math.log(rho_i / (r + sj)))
    elif rho_i < sj - r: # regime 3
        return 0.5 * (sj / (r**2 - sj**2) + 2/(sj - r) + (1.0/(2*r)) * math.log((sj - r)/(r + sj)))
    elif r + sj < rho_i: # no effect
        return 0.0

def test_generalized(r, rho_i, sj):
    if rho_i >= r + sj: return 0.0
    L = max(rho_i, r - sj)
    U = r + sj
    # wait I need to find the correct general formula! Let's check openmm via a curl to the github file and grep.
