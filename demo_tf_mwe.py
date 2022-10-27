import sys
from functools import partial

import gmsh
import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from petsc4py import PETSc

from mesh_sphere_axis import generate_mesh_sphere_axis


def curl_axis(a, m: int, rho):

    curl_r = (-a[2].dx(1) - 1j * m / rho * a[1])
    curl_p = (a[0].dx(1) - a[1].dx(0))

    return ufl.as_vector((curl_r, 0, curl_p))


def f_rz(x):

    a = x[0]/x[0]

    return (a, a)


def f_p(x):

    a = x[0]/x[0]

    return a


def pml_coordinate(
        x, r, alpha: float, k0: float, radius_dom: float, radius_pml: float):

    return (x + 1j * alpha / k0 * x * (r - radius_dom) / (radius_pml * r))


def create_eps_mu(pml, rho, eps_bkg, mu_bkg):

    J = ufl.grad(pml)

    # Transform the 2x2 Jacobian into a 3x3 matrix.
    J = ufl.as_matrix(((J[0, 0], J[0, 1], 0),
                       (J[1, 0], J[1, 1], 0),
                       (0, 0, pml[0] / rho)))

    A = ufl.inv(J)
    eps_pml = ufl.det(J) * A * eps_bkg * ufl.transpose(A)
    mu_pml = ufl.det(J) * A * mu_bkg * ufl.transpose(A)
    return eps_pml, mu_pml


omega_p = 30
gamma = 0.8

radius_sph = 0.002
radius_dom = 0.2
radius_scatt = 0.8 * radius_dom
radius_pml = 0.025

mesh_factor = 1
in_sph_size = mesh_factor * 0.08e-3
on_sph_size = mesh_factor * 0.06e-3
scatt_size = mesh_factor * 8.0e-3
pml_size = mesh_factor * 8.0e-3

tf_tag = 1
bkg_tag = 2
pml_tag = 3
hw_tag = 4  # tag for the hard-wall facet
scatt_tag = 5

model = None
gmsh.initialize(sys.argv)
if MPI.COMM_WORLD.rank == 0:

    model = generate_mesh_sphere_axis(
        radius_sph, radius_scatt, radius_dom, radius_pml,
        in_sph_size, on_sph_size, scatt_size, pml_size,
        tf_tag, bkg_tag, pml_tag, hw_tag, scatt_tag)

model = MPI.COMM_WORLD.bcast(model, root=0)

domain, cell_tags, facet_tags = model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()

degree = 3
curl_el = ufl.FiniteElement("N1curl", domain.ufl_cell(), degree)
div_el = ufl.FiniteElement("N1div", domain.ufl_cell(), degree)
lagr_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, ufl.MixedElement(
    [curl_el, lagr_el, div_el, lagr_el]))

dx = ufl.Measure("dx", domain, subdomain_data=cell_tags,
                 metadata={'quadrature_degree': 60})

dDom = dx((tf_tag, bkg_tag))
dPml = dx(pml_tag)

wl0 = 0.30  # Wavelength of the background field
k0 = 2 * np.pi / wl0  # Wavevector of the background field
omega0 = k0
theta = np.pi / 4  # Angle of incidence of the background field

m = 1  # list of harmonics

rho, z = ufl.SpatialCoordinate(domain)
alpha = 5
r = ufl.sqrt(rho**2 + z**2)

pml_coords = ufl.as_vector((
    pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
    pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

eps_pml, mu_pml = create_eps_mu(pml_coords, rho, 1, 1)

hw_facet = facet_tags.find(hw_tag)

div_space = V.sub(2).collapse()[0]

bc_dofs = fem.locate_dofs_topological(
    (V.sub(2), div_space), facet_tags.dim, hw_facet)

hw_bc = fem.Function(div_space)
with hw_bc.vector.localForm() as loc:
    loc.set(0)

bc = fem.dirichletbc(hw_bc, bc_dofs, V.sub(2))

dTf = dx(tf_tag)

E_func_space = fem.FunctionSpace(domain, ufl.MixedElement(
    [curl_el, lagr_el]))
P_func_space = fem.FunctionSpace(domain, ufl.MixedElement(
    [div_el, lagr_el]))

Es_rz_m, Es_p_m, P_rz_m, P_p_m = ufl.TrialFunctions(V)
v_rz_m, v_p_m, k_rz_m, k_p_m = ufl.TestFunctions(V)

Es_m = ufl.as_vector((Es_rz_m[0], Es_rz_m[1], Es_p_m))
P_m = ufl.as_vector((P_rz_m[0], P_rz_m[1], P_p_m))
v_m = ufl.as_vector((v_rz_m[0], v_rz_m[1], v_p_m))
k_m = ufl.as_vector((k_rz_m[0], k_rz_m[1], k_p_m))

Eb_m = fem.Function(E_func_space)
Eb_m.sub(0).interpolate(f_rz)
Eb_m.sub(1).interpolate(f_p)

curl_Es_m = curl_axis(Es_m, m, rho)
curl_v_m = curl_axis(v_m, m, rho)

F = + k0 ** 2 * ufl.inner(Es_m, v_m) * rho * dDom

# THIS TERM CAUSES THE PROBLEM!!!!
F += - ufl.inner(curl_Es_m, curl_v_m) * rho * dDom

F += + omega0 ** 2 * ufl.inner(P_m, k_m) * rho * dTf
F += + omega_p ** 2 * ufl.inner(Eb_m, k_m) * rho * dTf
F += - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml
F += + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

a, L = ufl.lhs(F), ufl.rhs(F)

problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={
    "ksp_type": "preonly", "pc_type": "asm", "sub_pc_type": "ilu"})

# Assemble lhs
problem._A.zeroEntries()

fem.petsc._assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
problem._A.assemble()

# Get diagonal of assembled A matrix
diagonal = problem._A.getDiagonal()
diagonal_values = diagonal.array
diagonal_size_arr = np.zeros(domain.comm.size, dtype=np.int32)
diagonal_size_arr[domain.comm.rank] = diagonal_values.size

domain.comm.Allreduce(MPI.IN_PLACE, diagonal_size_arr, op=MPI.SUM)
domain.comm.barrier()

# Get zero rows of assembled A matrix.
zero_rows = problem._A.findZeroRows()
zero_rows_values_global = zero_rows.array

# Maps global numbering to local numbering
zero_rows_values_local = zero_rows_values_global - \
    np.sum(diagonal_size_arr[:domain.comm.rank])

# Set diagonal entries of zero rows equal to one
diagonal_values[zero_rows_values_local] = 1
diagonal.array = diagonal_values
problem._A.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

# Assemble rhs
with problem._b.localForm() as b_loc:
    b_loc.set(0)
fem.petsc.assemble_vector(problem._b, problem._L)

# Apply boundary conditions to the rhs
fem.petsc.apply_lifting(problem._b, [problem._a], bcs=[problem.bcs])
problem._b.ghostUpdate(addv=PETSc.InsertMode.ADD,
                       mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(problem._b, problem.bcs)

# Solve linear system and update ghost values in the solution
problem._solver.solve(problem._b, problem._x)
problem.u.x.scatter_forward()

Esh_rz_m, Esh_p_m, Ph_rz_m, Ph_p_m = problem.u.split()

print(Esh_rz_m.x.array[:])

print(problem._solver.getConvergedReason())
