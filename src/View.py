'''
 # @ author: bcy | htc
 # @ date: 2024-05-20 16:53:59
 # @ license: MIT
 # @ description:  
 '''

import numpy as np
import pyvista as pv

from MyMesh import MyMesh
from Solver import Solver

def plotPointWithItsNeighbour(my_mesh: MyMesh, point_id: int) -> None:
    plotter = pv.Plotter()
    plotter.add_mesh(my_mesh.mesh, show_edges=True)
    point = my_mesh.mesh.points[point_id]
    for i_cell in my_mesh.cell_neighbour_ids_on_points[point_id]:
        plotter.add_mesh(my_mesh.mesh.extract_cells(i_cell), color="blue")
        pass
    plotter.add_mesh(pv.PolyData([point]), color="red", point_size=10)
    plotter.camera_position = "XY"
    plotter.show()
    pass

def markDirichletBoundary(mesh: MyMesh):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh.mesh, show_edges=True)
    plotter.add_mesh(pv.PolyData(mesh.mesh.points[list(mesh.dirichlet_boundary_dict.keys()), :]), color="red")
    plotter.camera_position = "XY"
    plotter.show()
    pass

def markNeumannBoundary(mesh: MyMesh):
    plotter = pv.Plotter()
    plotter.add_mesh(mesh.mesh, show_edges=True)
    for key in mesh.neumann_boundary_dict:
        point_id_1 = key[0]
        point_id_2 = key[1]
        point_1 = mesh.mesh.points[point_id_1]
        point_2 = mesh.mesh.points[point_id_2]
        plotter.add_mesh(pv.Line(point_1, point_2), color="blue", line_width=5)
        pass
    plotter.camera_position = "XY"
    plotter.show()
    pass

def plotCurrentStateOnPoints(solver: Solver):
    plotter = pv.Plotter()
    solver.combinePointDataToMesh()
    plotter.add_mesh(solver.mesh.mesh, scalars="u", show_edges=True, cmap="jet")
    plotter.camera_position = "XY"
    plotter.show()
    pass