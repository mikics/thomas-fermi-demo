import numpy
import numpy as np
import ufl
from dolfinx import cpp, fem, graph, io, mesh, plot
from mpi4py import MPI
from petsc4py.PETSc import ScalarType

rank = MPI.COMM_WORLD.rank

# Define mesh
domain = mesh.create_unit_square(
    MPI.COMM_WORLD, 8, 8, mesh.CellType.quadrilateral)

# Define function space
el1 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
el2 = ufl.FiniteElement("Lagrange", domain.ufl_cell(), 1)
mel = ufl.MixedElement([el1, el2])
Y = fem.FunctionSpace(domain, mel)

# Get subspaces
num_subs = Y.num_sub_spaces
spaces = []
maps = []
for i in range(num_subs):
    space_i, map_i = Y.sub(i).collapse()
    spaces.append(space_i)
    maps.append(map_i)
V = spaces[0]
Q = spaces[1]

# Create facet to cell connectivity required to determine boundary facets
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = numpy.flatnonzero(
    mesh.exterior_facet_indices(domain.topology))

# Finite Element problem
u, p = ufl.TrialFunctions(Y)
v, q = ufl.TestFunctions(Y)

f = fem.Constant(domain, ScalarType(-6))
a = ufl.dot(ufl.grad(u), ufl.grad(
    v)) * ufl.dx + ufl.dot(ufl.grad(p), ufl.grad(q)) * ufl.dx
L = f * v * ufl.dx + f * q * ufl.dx

# Define boundary conditions
uD = fem.Function(Y).sub(0)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
boundary_dofs_1 = fem.locate_dofs_topological(Y.sub(0), fdim, boundary_facets)
boundary_dofs_2 = fem.locate_dofs_topological(Y.sub(1), fdim, boundary_facets)
bc_1 = fem.dirichletbc(uD, boundary_dofs_1)
bc_2 = fem.dirichletbc(uD, boundary_dofs_2)
bc = [bc_1, bc_2]

# Solve variational problem
problem = fem.petsc.LinearProblem(a, L, bcs=bc, petsc_options={
                                  "ksp_type": "preonly", "pc_type": "lu"})
qh = problem.solve()

# Now, I want to change the solution values
q_n = fem.Function(Y)
q_n.sub(0).x.array[:] = -qh.sub(0).x.array
q_n.sub(1).x.array[:] = qh.sub(1).x.array*0

print(qh.x.array[:])
print(q_n.x.array[:])
