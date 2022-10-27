import sys
from functools import partial

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.special import jv, jvp, spherical_jn, spherical_yn

from mesh_sphere_axis import generate_mesh_sphere_axis

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)


def curl_axis(a, m: int, rho):

    curl_r = (-a[2].dx(1) - 1j * m / rho * a[1])
    curl_z = (a[2] / rho + a[2].dx(0) + 1j * m / rho * a[0])
    curl_p = (a[0].dx(1) - a[1].dx(0))

    return ufl.as_vector((curl_r, curl_z, curl_p))


def div_axis(a, m: int, rho):

    div_r = a[0] / rho + a[0].dx(0)
    div_z = a[1].dx(1)
    div_p = - 1j * m * a[2] / rho

    return div_r + div_z + div_p


def background_field_rz(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_r = (np.cos(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**(-m + 1) * jvp(m, k * x[0] * np.sin(theta), 1))

    a_z = (np.sin(theta) * np.exp(1j * k * x[1] * np.cos(theta))
           * (1j)**-m * jv(m, k * x[0] * np.sin(theta)))

    return (a_r, a_z)


def background_field_p(theta: float, n_bkg: float, k0: float, m: int, x):

    k = k0 * n_bkg

    a_p = (np.cos(theta) / (k * x[0] * np.sin(theta))
           * np.exp(1j * k * x[1] * np.cos(theta)) * m
           * (1j)**(-m) * jv(m, k * x[0] * np.sin(theta)))

    return a_p


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


epsilon_0 = 8.8541878128 * 10**-12  # Vacuum permittivity
mu_0 = 4 * np.pi * 10**-7  # Vacuum permeability
Z0 = np.sqrt(mu_0 / epsilon_0)  # Vacuum impedance
I0 = 0.5 / Z0  # Intensity of electromagnetic field
c0 = 1 / np.sqrt(mu_0 * epsilon_0)  # Speed of light in vacuum

_scaling_m = 1e6
_L_ref = 1/_scaling_m
_f_afac = _L_ref/c0
_beta_afac = 1/c0

n_bkg = 1
eps_bkg = n_bkg**2

omega_p = 9.1764e15*_f_afac
gamma = 2.4308e14*_f_afac
beta = 6.1776e5*_beta_afac

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

current_comm = "world"

if current_comm == "world":
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
    # Angular frequency of the background field
    omega0 = (k0 * _scaling_m * c0) * _f_afac
    theta = np.pi / 2  # Angle of incidence of the background field
    m_list = [1]  # list of harmonics

    rho, z = ufl.SpatialCoordinate(domain)
    alpha = 5
    r = ufl.sqrt(rho**2 + z**2)

    pml_coords = ufl.as_vector((
        pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
        pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

    eps_pml, mu_pml = create_eps_mu(pml_coords, rho, eps_bkg, 1)

    gcs = np.pi * radius_sph**2

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

    Eh_m = fem.Function(E_func_space)

    for m in m_list:

        Es_rz_m, Es_p_m, P_rz_m, P_p_m = ufl.TrialFunctions(V)
        v_rz_m, v_p_m, k_rz_m, k_p_m = ufl.TestFunctions(V)

        Es_m = ufl.as_vector((Es_rz_m[0], Es_rz_m[1], Es_p_m))
        P_m = ufl.as_vector((P_rz_m[0], P_rz_m[1], P_p_m))
        v_m = ufl.as_vector((v_rz_m[0], v_rz_m[1], v_p_m))
        k_m = ufl.as_vector((k_rz_m[0], k_rz_m[1], k_p_m))

        Eb_m = fem.Function(E_func_space)
        f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
        f_p = partial(background_field_p, theta, n_bkg, k0, m)
        Eb_m.sub(0).interpolate(f_rz)
        Eb_m.sub(1).interpolate(f_p)

        curl_Es_m = curl_axis(Es_m, m, rho)
        curl_v_m = curl_axis(v_m, m, rho)

        div_P_m = div_axis(P_m, m, rho)
        div_k_m = div_axis(k_m, m, rho)

        F = - ufl.inner(curl_Es_m, curl_v_m) * rho * dDom \
            + k0 ** 2 * ufl.inner(Es_m, v_m) * rho * dDom \
            + omega0 ** 2 * ufl.inner(P_m, v_m) * rho * dTf \
            - beta ** 2 * div_P_m * ufl.conj(div_k_m) * rho * dTf \
            + omega0 ** 2 * ufl.inner(P_m, k_m) * rho * dTf \
            + 1j * gamma * omega0 * ufl.inner(P_m, k_m) * rho * dTf \
            + omega_p ** 2 * ufl.inner(Es_m, k_m) * rho * dTf \
            + omega_p ** 2 * ufl.inner(Eb_m, k_m) * rho * dTf \
            - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml \
            + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

        a, L = ufl.lhs(F), ufl.rhs(F)

        problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={
                                          "ksp_type": "preonly", "pc_type": "lu"})

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
        zero_rows_values_local = zero_rows_values_global - \
            np.sum(diagonal_size_arr[:domain.comm.rank])

        print(f"rank: {domain.comm.rank}, local: {zero_rows_values_local}")
        print(f"rank: {domain.comm.rank}, global: {zero_rows_values_global}")

        # Set diagonal entries of zero rows equal to one
        diagonal_values[zero_rows_values_local] = 1
        diagonal.array = diagonal_values
        problem._A.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

        A_world = problem._A

        i_rows = range(np.sum(diagonal_size_arr[: domain.comm.rank]),
                       np.sum(diagonal_size_arr[: domain.comm.rank + 1])-1)

        i_columns = range(np.sum(diagonal_size_arr)-1)

        print(i_rows)
        print(i_columns)

        if domain.comm.rank == 0:
            vector_world_0 = A_world.getValues(i_rows, i_columns)

        if domain.comm.rank == 1:
            vector_world_1 = A_world.getValues(i_rows, i_columns)

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

        Esh_m = fem.Function(E_func_space)
        Esh_m.sub(0).interpolate(Esh_rz_m)
        Esh_m.sub(1).interpolate(Esh_p_m)

        Ph_m = fem.Function(P_func_space)
        Ph_m.sub(0).interpolate(Ph_rz_m)
        Ph_m.sub(1).interpolate(Ph_p_m)

        Jh_m = 1j * omega0 * Ph_m

        Eh_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

        Q = - 2 * np.pi * (ufl.inner(Eh_m, Jh_m))

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(2 * Q * rho * dTf)) / gcs).real
        print(q_abs_fenics_proc)
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

current_comm = "self"

if current_comm == "self" and MPI.COMM_WORLD.rank == 0:

    model = None
    gmsh.initialize(sys.argv)
    if MPI.COMM_WORLD.rank == 0:

        model = generate_mesh_sphere_axis(
            radius_sph, radius_scatt, radius_dom, radius_pml,
            in_sph_size, on_sph_size, scatt_size, pml_size,
            tf_tag, bkg_tag, pml_tag, hw_tag, scatt_tag)

    model = MPI.COMM_WORLD.bcast(model, root=0)

    domain, cell_tags, facet_tags = model_to_mesh(
        model, MPI.COMM_SELF, 0, gdim=2)

    gmsh.finalize()

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
    # Angular frequency of the background field
    omega0 = (k0 * _scaling_m * c0) * _f_afac
    theta = np.pi / 2  # Angle of incidence of the background field
    m_list = [1]  # list of harmonics

    rho, z = ufl.SpatialCoordinate(domain)
    alpha = 5
    r = ufl.sqrt(rho**2 + z**2)

    pml_coords = ufl.as_vector((
        pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
        pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

    eps_pml, mu_pml = create_eps_mu(pml_coords, rho, eps_bkg, 1)

    gcs = np.pi * radius_sph**2

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

    Eh_m = fem.Function(E_func_space)

    for m in m_list:

        Es_rz_m, Es_p_m, P_rz_m, P_p_m = ufl.TrialFunctions(V)
        v_rz_m, v_p_m, k_rz_m, k_p_m = ufl.TestFunctions(V)

        Es_m = ufl.as_vector((Es_rz_m[0], Es_rz_m[1], Es_p_m))
        P_m = ufl.as_vector((P_rz_m[0], P_rz_m[1], P_p_m))
        v_m = ufl.as_vector((v_rz_m[0], v_rz_m[1], v_p_m))
        k_m = ufl.as_vector((k_rz_m[0], k_rz_m[1], k_p_m))

        Eb_m = fem.Function(E_func_space)
        f_rz = partial(background_field_rz, theta, n_bkg, k0, m)
        f_p = partial(background_field_p, theta, n_bkg, k0, m)
        Eb_m.sub(0).interpolate(f_rz)
        Eb_m.sub(1).interpolate(f_p)

        curl_Es_m = curl_axis(Es_m, m, rho)
        curl_v_m = curl_axis(v_m, m, rho)

        div_P_m = div_axis(P_m, m, rho)
        div_k_m = div_axis(k_m, m, rho)

        F = - ufl.inner(curl_Es_m, curl_v_m) * rho * dDom \
            + k0 ** 2 * ufl.inner(Es_m, v_m) * rho * dDom \
            + omega0 ** 2 * ufl.inner(P_m, v_m) * rho * dTf \
            - beta ** 2 * div_P_m * ufl.conj(div_k_m) * rho * dTf \
            + omega0 ** 2 * ufl.inner(P_m, k_m) * rho * dTf \
            + 1j * gamma * omega0 * ufl.inner(P_m, k_m) * rho * dTf \
            + omega_p ** 2 * ufl.inner(Es_m, k_m) * rho * dTf \
            + omega_p ** 2 * ufl.inner(Eb_m, k_m) * rho * dTf \
            - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml \
            + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

        a, L = ufl.lhs(F), ufl.rhs(F)

        problem = fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={
            "ksp_type": "preonly", "pc_type": "lu"})

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
        zero_rows_values_local = zero_rows_values_global - \
            np.sum(diagonal_size_arr[:domain.comm.rank])

        print(f"rank: {domain.comm.rank}, local: {zero_rows_values_local}")
        print(f"rank: {domain.comm.rank}, global: {zero_rows_values_global}")

        # Set diagonal entries of zero rows equal to one
        diagonal_values[zero_rows_values_local] = 1
        diagonal.array = diagonal_values
        problem._A.setDiagonal(diagonal, PETSc.InsertMode.INSERT_VALUES)

        A_self = problem._A

        print(i_columns)
        vector_self = A_self.getValues(i_columns, i_columns)

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

        Esh_m = fem.Function(E_func_space)
        Esh_m.sub(0).interpolate(Esh_rz_m)
        Esh_m.sub(1).interpolate(Esh_p_m)

        Ph_m = fem.Function(P_func_space)
        Ph_m.sub(0).interpolate(Ph_rz_m)
        Ph_m.sub(1).interpolate(Ph_p_m)

        Jh_m = 1j * omega0 * Ph_m

        Eh_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

        Q = - 2 * np.pi * (ufl.inner(Eh_m, Jh_m))

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(2 * Q * rho * dTf)) / gcs).real
        print(q_abs_fenics_proc)
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)