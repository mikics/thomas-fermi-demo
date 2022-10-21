import sys
from functools import partial

import numpy as np
import ufl
from dolfinx import fem, mesh, plot
from dolfinx.io import VTXWriter
from dolfinx.io.gmshio import model_to_mesh
from mpi4py import MPI
from petsc4py import PETSc
from scipy.special import jv, jvp

from mesh_sphere_axis import generate_mesh_sphere_axis

try:
    import gmsh
except ModuleNotFoundError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)

try:
    import pyvista
    have_pyvista = True
except ModuleNotFoundError:
    print("pyvista and pyvistaqt are required to visualise the solution")
    have_pyvista = False
if not np.issubdtype(PETSc.ScalarType, np.complexfloating):
    print("Demo should only be executed with DOLFINx complex mode")
    exit(0)


def curl_axis(a, m: int, rho):

    curl_r = -a[2].dx(1) - 1j * m / rho * a[1]
    curl_z = a[2] / rho + a[2].dx(0) + 1j * m / rho * a[0]
    curl_p = a[0].dx(1) - a[1].dx(0)

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

n_bkg = 1
eps_bkg = n_bkg**2

omega_p = 9.1764e15
gamma = 2.4308e14
beta = 6.1776e5

radius_sph = 0.002
radius_dom = 0.2
radius_scatt = 0.4 * radius_dom
radius_pml = 0.025

mesh_factor = 1
in_sph_size = mesh_factor * 0.08e-3
on_sph_size = mesh_factor * 0.06e-3
scatt_size = mesh_factor * 10.0e-3
pml_size = mesh_factor * 8.0e-3

au_tag = 1
bkg_tag = 2
pml_tag = 3
hw_tag = 4
scatt_tag = 5

model = None
gmsh.initialize(sys.argv)
if MPI.COMM_WORLD.rank == 0:

    model = generate_mesh_sphere_axis(
        radius_sph, radius_scatt, radius_dom, radius_pml,
        in_sph_size, on_sph_size, scatt_size, pml_size,
        au_tag, bkg_tag, pml_tag, hw_tag, scatt_tag)

model = MPI.COMM_WORLD.bcast(model, root=0)
domain, cell_tags, facet_tags = model_to_mesh(
    model, MPI.COMM_WORLD, 0, gdim=2)

gmsh.finalize()
MPI.COMM_WORLD.barrier()

degree = 3
curl_el = ufl.FiniteElement("N1curl", domain.ufl_cell(), degree)
div_el = ufl.FiniteElement("N1div", domain.ufl_cell(), degree-1)
lagr_el = ufl.FiniteElement("Lagrange", domain.ufl_cell(), degree)
V = fem.FunctionSpace(domain, ufl.MixedElement(
    [curl_el, lagr_el, div_el, lagr_el]))

dx = ufl.Measure("dx", domain, subdomain_data=cell_tags,
                 metadata={'quadrature_degree': 60})

dDom = dx((au_tag, bkg_tag))
dPml = dx(pml_tag)

wl0 = 0.4  # Wavelength of the background field
k0 = 2 * np.pi / wl0  # Wavevector of the background field
c0 = 1 / np.sqrt(epsilon_0 * mu_0)
omega0 = k0 * c0  # Angular frequency of the background field
theta = np.pi / 4  # Angle of incidence of the background field
m_list = [0, 1]  # list of harmonics

rho, z = ufl.SpatialCoordinate(domain)
alpha = 5
r = ufl.sqrt(rho**2 + z**2)

pml_coords = ufl.as_vector((
    pml_coordinate(rho, r, alpha, k0, radius_dom, radius_pml),
    pml_coordinate(z, r, alpha, k0, radius_dom, radius_pml)))

eps_pml, mu_pml = create_eps_mu(pml_coords, rho, 1, 1)

gcs = np.pi * radius_sph**2

hw_facet = facet_tags.find(hw_tag)

div_space = V.sub(2).collapse()[0]

bc_dofs = fem.locate_dofs_topological(div_space, facet_tags.dim, hw_facet)

u_bc = fem.Function(V)
with u_bc.vector.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

dAu = dx(au_tag)

for m in m_list:

    Es_rz_m, Es_p_m, P_rz_m, P_p_m = ufl.TrialFunctions(V)
    v_rz_m, v_p_m, k_rz_m, k_p_m = ufl.TestFunctions(V)

    Es_m = ufl.as_vector((Es_rz_m[0], Es_rz_m[1], Es_p_m))
    P_m = ufl.as_vector((P_rz_m[0], P_rz_m[1], P_p_m))
    v_m = ufl.as_vector((v_rz_m[0], v_rz_m[1], v_p_m))
    k_m = ufl.as_vector((k_rz_m[0], k_rz_m[1], k_p_m))

    E_func_space = fem.FunctionSpace(domain, ufl.MixedElement(
        [curl_el, lagr_el]))
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
        + omega0 ** 2 * mu_0 * ufl.inner(P_m, v_m) * rho * dAu \
        - beta ** 2 * div_P_m * ufl.conj(div_k_m) * rho * dAu \
        + omega0 ** 2 * ufl.inner(P_m, k_m) * rho * dAu \
        + 1j * gamma * omega0 * ufl.inner(P_m, k_m) * rho * dAu \
        + epsilon_0 * omega_p ** 2 * ufl.inner(Es_m, k_m) * rho * dAu \
        + epsilon_0 * omega_p ** 2 * ufl.inner(Eb_m, k_m) * rho * dAu \
        - ufl.inner(ufl.inv(mu_pml) * curl_Es_m, curl_v_m) * rho * dPml \
        + k0 ** 2 * ufl.inner(eps_pml * Es_m, v_m) * rho * dPml

    a, L = ufl.lhs(F), ufl.rhs(F)

    # a.ident_zeros()

    problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={
                                      "ksp_type": "preonly", "pc_type": "lu"})
    # Assemble lhs
    problem._A.zeroEntries()

    fem.petsc._assemble_matrix_mat(problem._A, problem._a, bcs=problem.bcs)
    problem._A.assemble()
    diagonal = problem._A.getDiagonal()
    zero_rows = problem._A.findZeroRows()
    vals = diagonal.array
    vals[zero_rows] = 1
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

    Esh_m = fem.Function(E_func_space)
    Esh_m.sub(0).interpolate(Esh_rz_m)
    Esh_m.sub(1).interpolate(Esh_p_m)

    P_func_space = fem.FunctionSpace(domain, ufl.MixedElement(
        [div_el, lagr_el]))
    Ph_m = fem.Function(P_func_space)
    Ph_m.sub(0).interpolate(Ph_rz_m)
    Ph_m.sub(1).interpolate(Ph_p_m)

    Jh_m = 1j * omega0 * Ph_m

    Eh_m = fem.Function(E_func_space)

    Eh_m.x.array[:] = Eb_m.x.array[:] + Esh_m.x.array[:]

    if m == 0:  # initialize and do not add 2 factor
        Q = - np.pi * (ufl.inner(Eh_m, Jh_m))

        q_abs_fenics_proc = (fem.assemble_scalar(
                             fem.form(Q * rho * dAu)) / gcs / I0).real
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

    elif m == m_list[0]:  # initialize and add 2 factor
        Q = - 2 * np.pi * (ufl.inner(Eh_m, Jh_m))

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(Q * rho * dAu)) / gcs / I0).real
        q_abs_fenics = domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

    else:  # do not initialize and add 2 factor
        Q += - 2 * np.pi * (ufl.inner(Eh_m, Jh_m))

        q_abs_fenics_proc = (fem.assemble_scalar(
            fem.form(Q * rho * dAu)) / gcs / I0).real
        q_abs_fenics += domain.comm.allreduce(q_abs_fenics_proc, op=MPI.SUM)

print(q_abs_fenics)
sys.exit()

q_ext_fenics = q_abs_fenics + q_sca_fenics

q_abs_analyt = 0.9622728008329892
q_sca_analyt = 0.07770397394691526
q_ext_analyt = q_abs_analyt + q_sca_analyt

err_abs = np.abs(q_abs_analyt - q_abs_fenics) / q_abs_analyt
err_sca = np.abs(q_sca_analyt - q_sca_fenics) / q_sca_analyt
err_ext = np.abs(q_ext_analyt - q_ext_fenics) / q_ext_analyt

if MPI.COMM_WORLD.rank == 0:

    print()
    print(f"The analytical absorption efficiency is {q_abs_analyt}")
    print(f"The numerical absorption efficiency is {q_abs_fenics}")
    print(f"The error is {err_abs*100}%")
    print()
    print(f"The analytical scattering efficiency is {q_sca_analyt}")
    print(f"The numerical scattering efficiency is {q_sca_fenics}")
    print(f"The error is {err_sca*100}%")
    print()
    print(f"The analytical extinction efficiency is {q_ext_analyt}")
    print(f"The numerical extinction efficiency is {q_ext_fenics}")
    print(f"The error is {err_ext*100}%")

    # Check whether the geometrical or optical parameters ar correct
    assert radius_sph / wl0 == 0.025 / 0.4
    assert eps_au == -1.0782 + 1j * 5.8089

    assert err_abs < 0.01
    assert err_sca < 0.01
    assert err_ext < 0.01

Esh_rz, Esh_p = Esh.split()

Esh_rz_dg = fem.Function(V_dg)
Esh_r_dg = fem.Function(V_dg)

Esh_rz_dg.interpolate(Esh_rz)

with VTXWriter(domain.comm, "sols/Es_rz.bp", Esh_rz_dg) as f:
    f.write(0.0)
with VTXWriter(domain.comm, "sols/Es_p.bp", Esh_p) as f:
    f.write(0.0)

if have_pyvista:
    V_cells, V_types, V_x = plot.create_vtk_mesh(V_dg)
    V_grid = pyvista.UnstructuredGrid(V_cells, V_types, V_x)
    Esh_r_values = np.zeros((V_x.shape[0], 3), dtype=np.float64)
    Esh_r_values[:, :domain.topology.dim] = \
        Esh_r_dg.x.array.reshape(V_x.shape[0], domain.topology.dim).real

    V_grid.point_data["u"] = Esh_r_values

    pyvista.set_jupyter_backend("pythreejs")
    plotter = pyvista.Plotter()

    plotter.add_text("magnitude", font_size=12, color="black")
    plotter.add_mesh(V_grid.copy(), show_edges=False)
    plotter.view_xy()
    plotter.link_views()

    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        pyvista.start_xvfb()
        plotter.screenshot("Esh_r.png", window_size=[500, 500])
