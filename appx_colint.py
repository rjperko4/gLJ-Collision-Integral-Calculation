"""
This script takes an input (see readme) and interpolates the collision integrals omega11 and omega22 for given compounds.
This is a very quick, rough linear interpolation of the collision integrals, which is especially inaccurate when the values
become erratic around beta < 2. For a slower, accurate calculation use collison_integrals.py

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
"""
import sys, os
import numpy as np
import scipy.interpolate as interp
from scipy.optimize import newton, brenth


def poly_sigma_star(x, beta):
    """
    Polynomial function for which sigma_star is the positive root
    args:
        beta: float, value of beta
    """
    return 1 - x**6 - (6 / beta) * x ** (12 - beta)


def poly_sigma_star_prime(x, beta):
    """
    Derivative of poly_sigma_star
    args:
        x: float, value of x
        beta: float, value of beta
    """
    return -6 * x**5 - (12 - beta) * (6 / beta) * x ** (11 - beta)


def find_sigma_star(beta):
    """
    Finds the value of sigma_star, the ratio sigma/r_m.
    args:
        beta: float, value of beta
    """
    if beta > 14.68422:
        print(f"ERROR: Beta value {beta} is too high for this method.")
        exit()

    root = newton(poly_sigma_star, x0=1, fprime=poly_sigma_star_prime, args=(beta,))

    # If beta > 12, must look for largest root
    if beta > 12:
        if (
            root < 0.8223
        ):  # Basically just imperically determined the smallest possible value of the largest root
            root = brenth(poly_sigma_star, root + 1e-6, root + 1, args=(beta))

    if root < 0:
        root = newton(poly_sigma_star, x0=2, fprime=poly_sigma_star_prime, args=(beta,))
        if root < 0:
            print(f"ERROR: Root {root} is negative for beta = {beta}")
            exit()

    return root


def interp_colint(ls, T_star, beta):
    infile = "colint_11_table.csv" if ls == 11 else "colint_22_table.csv"
    with open(infile) as IN:
        header = IN.readline().strip().split(",")
        beta_values = [float(x) for x in header[1:]]
        table = np.loadtxt(IN, delimiter=",")
        T_star_values = table[:, 0]
        table = table[:, 1:]

    X = []
    Y = []
    Z = []
    for i in range(len(T_star_values)):
        for j in range(len(beta_values)):
            X.append(T_star_values[i])
            Y.append(beta_values[j])
            Z.append(table[i, j])
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)
    if T_star < min(T_star_values) or T_star > max(T_star_values):
        print("Error: T_star out of data range")
        sys.exit(1)
    if beta < min(beta_values) or beta > max(beta_values):
        print("Error: beta out of data range")
        sys.exit(1)

    return interp.griddata((X, Y), Z, (T_star, beta), method="linear")


def viscosity(M, T, sigma, omega):
    """
    Calculates the viscosity of a fluid
    args:
        M: float, molecular weight
        T: float, temperature
        sigma: float, collision diameter
        omega: float, collision integral (l=2,s=2)
    """
    return (2.6693e-5) * np.sqrt(M * T) / (sigma**2 * omega)


def diffusion(M, T, P, sigma, omega):
    """
    Calculates the diffusion coefficient of a fluid
    args:
        M: float, molecular weight
        T: float, temperature
        P: float, pressure
        sigma: float, collision diameter
        omega: float, collision integral (l=1,s=1)
    """
    return (0.002628) * np.sqrt(T**3 / M) / (P * sigma**2 * omega)


def thermal_conductivity(M, T, sigma, omega):
    """
    Calculates the thermal conductivity of a fluid
    args:
        M: float, molecular weight
        T: float, temperature
        sigma: float, collision diameter
        omega: float, collision integral (l=2,s=2)
    """
    return (1.9891e-4) * np.sqrt(T / M) / (sigma**2 * omega)


def main():
    if len(sys.argv) < 2:
        name = os.path.basename(__file__)
        exit(f"usage: python3 {name} <input> [output]")
    in_file = sys.argv[1]
    out_file = sys.argv[2] if len(sys.argv) > 2 else "output.csv"
    with open(in_file, "r") as f:
        data = [line.split(sep=",") for line in f]
        data = data[1:]  # Remove header
    if os.path.exists(out_file):
        resp = input("Output file already exists. Overwrite? Y/N: ")
        if resp.lower() == "y" or resp.lower() == "yes":
            os.remove(out_file)
        else:
            exit()

    with open(out_file, "w") as f:
        f.write(
            """compound, epsilon/k (K), beta, T (K), P (atm), omega11, omega22, diffusion (cm^2/s), viscosity (g/(cm*s)), diffusion error, viscosity error\n"""
        )

    for row in data:
        compound = row[0]
        eps_k = float(row[1])  # epsilon/k
        if row[2].strip() == "":
            if row[3] == "":
                exit("ERROR: Must provide either r_m or sigma_LJ")
            sigma_LJ = float(row[3])
            r_m = sigma_LJ * 2 ** (1 / 6)  # standard LJ relationship
        else:
            r_m = float(row[2])
        beta = float(row[4]) if row[4] != "" else 6
        sigma = r_m * find_sigma_star(beta)
        T = float(row[5])
        M = float(row[6])
        P = float(row[7])
        v_benchmark = float(row[8]) if row[8].strip() != "" else "None"
        d_benchmark = float(row[9]) if row[9].strip() != "" else None
        omega11 = interp_colint(11, T / eps_k, beta)
        omega22 = interp_colint(22, T / eps_k, beta)
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
        with open(out_file, "a") as f:
            f.write(
                f"{compound}, {eps_k}, {beta}, {T}, {P}, {omega11}, {omega22}, {diff}, {vis}, {diff_error}, {vis_error}\n"
            )
    print(f"Output written to {out_file}")


if __name__ == "__main__":
    main()
