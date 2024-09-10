"""
This script takes an input (see readme) and calculates the collision integrals
omega11 and omega22 for given compounds.For a much faster approximation, use 
the same input file with appx_colint.py.

As mentioned in the associated paper, it is not uncommon to raise some 
non-fatal integration warnings. These have not significantly impacted the 
results in testing.

Expected units for input variables:

epsilon/k: K
r_m: angstroms (optional if sigma_LJ is provided, leave blank)
sigma_LJ: angstroms (optional if r_m is provided, leave blank)
beta: unitless
T: K
M: g/mol
P: atm
viscosity benchmark: g/(cm*s)
diffusion benchmark: cm^2/s

A note on variables:
All calculations are done with 'reduced' variables, described in the 
accompanying paper. These are typically denoted with a * (for example, 
r* = r/r_m). For the sake of consiceness, however, these variables are not 
denoted with a * in the code (with the exception of T_star and sigma_star, here
they must be disinguished from their unscaled counterparts). So the following
variables actually represent their reduced counterparts:

r -> r* = r/r_m
b -> b* = b/r_m
g -> g* = sqrt((mg^2/4)/(4*epsilon))
Q -> Q* = Q/Q_rigid
omega  -> Omega* = Omega/Omega_rigid
"""

import sys, os
from math import factorial
import numpy as np
from scipy.optimize import newton, brenth
from scipy.integrate import quad

### Mathematical Function Definitions ###
def func_sigma_star(x, beta):
    """
    function function of which sigma_star is the positive root
    args:
        beta: float, value of beta
    """
    return 1 - x**6 - (6/beta)*x**(12-beta)

def func_sigma_star_prime(x, beta):
    """
    Derivative of func_sigma_star
    args:
        x: float, value of x
        beta: float, value of beta
    """
    return -6*x**5 - (12-beta)*(6/beta)*x**(11-beta)

def func_A(x, g, b, beta):
    """
    function function of which A is the smallest positive root
    args:
        x: float, value of x
        g: float, relative velocity before interaction
        b: float, impact parameter of collision
        beta: float, value of beta
    """
    return -(beta/(6*g**2))*(x**12 - x**6 - (6/beta)*x**beta) - b**2*x**2 + 1

def func_A_prime(x, g, b, beta):
    """
    Derivative of func_A
    args:
        x: float, value of x
        g: float, relative velocity before interaction
        b: float, impact parameter of collision
        beta: float, value of beta
    """
    if x == 0:
        return func_A_prime(1e-16, g, b, beta)
    return  -(beta/(6*g**2))*(12*x**11 - 6*x**5 - 6*x**(beta-1)) - 2*b**2*x

def func_A_double_prime(x, g, b, beta):
    """
    Second derivative of func_A
    args:
        x: float, value of x
        g: float, relative velocity before interaction
        b: float, impact parameter of collision
        beta: float, value of beta
    """
    if x == 0:
        return func_A_double_prime(1e-16, g, b, beta)
    return  -(beta/(6*g**2))*(132*x**10 - 30*x**4 - (beta-1)*6*x**(beta-2)) - 2*b**2

def theta_m_integrand(rho, g, b, beta, A):
    """
    The function to be integrated to find theta_m.
    args:
        x: float, values of x
        g: float, relative velocity before interaction
        b: float, impact parameter of collision
        beta: float, value of beta
    """
    d = 6/beta
    integrand = (1 - b**2*rho**2 - (rho**12 - rho**6 - d*rho**beta)/(d*g**2))
    if integrand <= 0:
        if integrand > -1e-6: # Rounding error in calculating the integrand
            integrand = 1e-16
        elif abs(rho-A) < 1e-9: # Rounding error in calculating A
            integrand = 1e-16
        else:
            # Negative results should only be rounding errors. If you have a
            # substantial negative result, there is a mathematical error.
            print(f"""ERROR: Negative theta_m integrand: rho = {rho}, b = {b},
                g = {g}, beta = {beta}, A = {A}, integrand = {integrand}""")
            print("""This is likely do to an error in finding A. Make sure A is
                the smallest root of the function.""")
            sys.exit()

    try:
        integrand = integrand**(-0.5)
    except:
        print(f"""ERROR2: Negative theta_m integrand: rho = {rho}, b = {b},
            g = {g}, beta = {beta}, A = {A}, integrand = {integrand}""")
        print("""This is likely do to an error in finding A. Make sure A
            is the smallest root of the function.""")
        exit()
    return integrand

def Q_integrand(b, g, beta, l):
    """
    The function to be integrated to find Q.
    args:
        b: float, impact parameter of collision
        g: float, relative velocity before interaction
        beta: float, value of beta
        l: int, only 1 or 2 for first-order approximations
    """
    theta_m = find_theta_m(b, g, beta)
    integrand = (1 - (-1)**l * (np.cos(2*theta_m) ** l)) * b
    return integrand

def omega_integrand(g, beta, sigma_star, T_star, l, s):
    """
    The function to be integrated to find Q.
    args:
        g: float, relative velocity before interaction
        beta: float, value of beta
        sigma_star: float, sigma/r_m
        T: float, temperature
        l: int, only 1 or 2 for first-order approximations
        s: int, only 1 or 2 for first-order approximations
    """
    a = np.exp(-g**2 / T_star)
    Q = find_Q(g, beta, sigma_star, l)
    integrand = a * g**(2*s+3) * Q
    print("*",end="",flush=True)
    return integrand

### Methods ###
def find_sigma_star(beta):
    """
    Finds the sigma/r_m, the ratio sigma/r_m.
    args:
        beta: float, value of beta
    """
    if beta > 14.68422:
        print(f"ERROR: Beta value {beta} is too high for this method.")
        sys.exit()

    root = newton(func_sigma_star, x0 = 1, fprime=func_sigma_star_prime, args=(beta,))

    # If beta > 12, must look for largest root
    if beta > 12:
        if root < 0.8223: # Imperically determined smallest possible 2nd root
            root = brenth(func_sigma_star, root + 1e-6, root + 1, args=(beta))

    if root < 0:
        root = newton(func_sigma_star, x0 = 2, fprime=func_sigma_star_prime, args=(beta,))
        if root < 0:
            print(f"ERROR: Root {root} is negative for beta = {beta}")
            sys.exit()
    return root

def find_A(b,g,beta):
    """
    Finds the value of A by finding the smallest positive root of the function func_A.
    Includes checks to ensure the root is positive and the smallest root.
    args:
        b: float, impact parameter of collision
        g: float, relative velocity before interaction
        beta: float, value of beta
    """

    # Find first guess as accurately as possible. The first root should be 
    # preceeded by positive values
    x0 = 0
    start = 0
    step_size = 0.001
    for _ in range(12):
        x0 = start
        px0 = func_A(x0, g, b, beta)
        dp_sign = np.sign(func_A_prime(x0, g, b, beta))
        while px0 > 0:
            #If the slope changes from negative to positive then check minima
            if func_A_prime(x0, g, b, beta) * dp_sign < 0:
                # This method searches for double roots that dip below the 
                # x-axis for only a small interval
                if dp_sign < 0:
                    if x0 != start:
                        try:
                            dx_root = brenth(func_A_prime, x0 - step_size, x0,
                                args=(g, b, beta))
                        except:
                            print(f"""ERROR dx_root: b = {b}, g = {g}, beta = 
                                {beta}, x0 = {x0}, dp_sign = {dp_sign}""")
                            sys.exit()
                        if func_A(dx_root, g, b, beta) <= 0:
                            x0 = dx_root
                            break
                        dp_sign *= -1
                else:
                    dp_sign *= -1
            x0 += step_size
            px0 = func_A(x0, g, b, beta)
        if x0 >= start + step_size:
            start = x0 - step_size
        step_size /= 10
    # Ensures no negative evaluations by rounding error
    while func_A(start, g, b, beta) < 0:
        start -= 1e-16
    return x0

def find_theta_m(b,g,beta):
    """
    Calculates the integral that defines theta_m
        b: float, impact parameter of collision
        g: float, relative velocity before interaction
        beta: float, value of beta
    """
    A = find_A(b, g, beta)
    result = quad(lambda rho: theta_m_integrand(rho, g, b, beta, A), 0, A, full_output=1)
    integral = result[0]
    theta_m = b * integral
    return theta_m

def find_Q(g, beta, sigma_star, l):
    """
    Calculates the integral that defines Q
        b: float, impact parameter of collision
        g: float, relative velocity before interaction
        beta: float, value of beta
        sigma_star: float, sigma/r_m
        l: int, only 1 or 2 for first-order approximations
    """
    if type(l) != int or l < 0:
        print(f"ERROR: l must be an integer, not {l}")
        sys.exit()
    integral, _ = quad(lambda b: Q_integrand(b, g, beta, l), 0, np.inf)
    a = 1/(1 - 0.5 * ((1 + (-1)**l)/(1+l)))
    Q = 2 * a * integral / sigma_star**2
    return Q

def find_Omega(beta, sigma_star, T_star, l, s):
    """
    Calculates omega including the integral 
        beta: float, value of beta
        sigma_star: float, sigma/r_m
        T: float, temperature
        l: int, only 1 or 2 for first-order approximations
        s: int, only 1 or 2 for first-order approximations
    """
    if type(l) != int or l < 0:
        print(f"ERROR: l must be an integer, not {l}")
        exit()
    if type(s) != int or s < 0:
        print(f"ERROR: s must be an integer, not {s}")
        sys.exit()
    integral, _ = quad(lambda g: omega_integrand(g, beta, sigma_star, T_star, l, s), 0, np.inf)
    a = 2/(factorial(s+1) * T_star**(s+2))
    omega  = a * integral
    return omega

def diffusion(M,T,P,sigma,omega):
    """
    Calculates the self-diffusion coefficient of a pure gas
    args:
        M: float, molecular weight
        T: float, temperature
        P: float, pressure
        sigma: float, collision diameter 
        omega: float, collision integral (l=1,s=1)
    """
    return (0.002628)*np.sqrt(T**3/M)/(P * sigma**2 * omega)

def viscosity(M,T,sigma,omega):
    """
    Calculates the viscosity of a pure gas
    args:
        M: float, molecular weight
        T: float, temperature
        sigma: float, collision diameter 
        omega: float, collision integral (l=2,s=2)
    """
    return (2.6693e-5)*np.sqrt(M * T)/(sigma**2 * omega)

def thermal_conductivity(M,T,sigma,omega):
    """
    (UNTESTED) Calculates the thermal conductivity of a pure gas 
    args:
        M: float, molecular weight
        T: float, temperature
        sigma: float, collision diameter 
        omega: float, collision integral (l=2,s=2)
    """
    return (1.9891e-4)*np.sqrt(T/M)/(sigma**2 * omega)

def main():
    if len(sys.argv) < 2:
        name = os.path.basename(__file__)
        sys.exit(f"usage: python3 {name} <input> [output]")
    in_file = sys.argv[1] 
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    with open(in_file, 'r') as f:
        data = [line.split(sep=',') for line in f]
        data = data[1:] # Remove header
    if os.path.exists(out_file):
        resp = input("Output file already exists. Overwrite? Y/N: ")
        if resp.lower() == 'y' or resp.lower() == 'yes':
            os.remove(out_file)
        else:
            sys.exit()


    with open(out_file, 'w') as f:
        f.write("compound, eps/k, beta, T (K), P (atm), omega11, omega22, ")
        f.write("diffusion (cm^2/s), viscosity (g/(cm*s)), diffusion error, ")
        f.write("viscosity error\n")


    for row in data:
        compound = row[0]
        eps_k = float(row[1]) # epsilon/k
        if row[2] == "":
            if row[3] == "":
                sys.exit("ERROR: Must provide either r_m or sigma_LJ")
            sigma_LJ = float(row[3])
            r_m = sigma_LJ * 2**(1/6) # standard LJ relationship
        else:
            r_m = float(row[2])
        beta = float(row[4]) if row[4] != "" else 6
        T = float(row[5])
        M = float(row[6])
        P = float(row[7])
        v_benchmark = float(row[8]) if row[8].strip() != "" else None
        d_benchmark = float(row[9]) if row[9].strip() != "" else None

        sigma_star = find_sigma_star(beta)
        sigma = sigma_star * r_m
        T_star = T/eps_k
        omega11 = find_Omega(beta, sigma_star, T_star, 1, 1)
        omega22 = find_Omega(beta, sigma_star, T_star, 2, 2)
        diff = diffusion(M, T, P, sigma, omega11)
        vis = viscosity(M, T, sigma, omega22)
        if v_benchmark is not None:
            vis_error = abs(v_benchmark - vis) / v_benchmark
        else:
            vis_error = np.nan
        if d_benchmark is not None:
            diff_error = abs(d_benchmark - diff) / d_benchmark
        else:
            diff_error = np.nan
        with open(out_file, 'a') as f:
            f.write(f"{compound}, {eps_k}, {beta}, {T}, {P}, {omega11}, ")
            f.write(f"{omega22}, {diff}, {vis}, {diff_error}, {vis_error}\n")
    print(f"Output written to {out_file}")

if __name__ == "__main__":
    main()