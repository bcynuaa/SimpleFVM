'''
 # @ author: bcy | htc
 # @ date: 2024-05-20 15:47:32
 # @ license: MIT
 # @ description:  
 '''

import numpy as np
import pyvista as pv

class MyMesh:

    def __init__(self) -> None:
        # dirichlet_boundary_dict = {point_id: value}
        self.dirichlet_boundary_dict: dict = dict()
        # neumann_boundary_dict = {(point_id_1, point_id_2): value}
        self.neumann_boundary_dict: dict = dict()
        pass

    def __call__(self) -> pv.UnstructuredGrid:
        return self.mesh
        pass

    def readCGNS(self, mesh_filename, x_index: int = 0, y_index = 1) -> None:
        self.mesh_filename: str = mesh_filename
        self.mesh: pv.UnstructuredGrid = pv.read(self.mesh_filename).combine()
        self.points: np.ndarray = np.copy(self.mesh.points)
        self.points[:, 0] = self.points[:, x_index]
        self.points[:, 1] = self.points[:, y_index]
        self.points[:, 2] = 0.0
        self.mesh.points = self.points
        self.preprocessMesh()
        pass

    def preprocessMesh(self) -> None:
        self.n_points: int = self.mesh.n_points
        self.n_cells: int = self.mesh.n_cells
        self.mesh = self.mesh.extract_cells(np.arange(self.n_cells))
        self.mesh = self.mesh.compute_cell_sizes(area=True, volume=False, length=False)
        self.area: np.ndarray = np.array(self.mesh.cell_data["Area"])
        self.__getCellNeighbourIdsOnPoints()
        pass
        
    def __getCellNeighbourIdsOnPoints(self) -> None:
        self.cell_neighbour_ids_on_points: list[list] = [list() for _ in range(self.n_points)]
        for i_cell in range(self.n_cells):
            current_cell_id: int = i_cell
            current_cell = self.mesh.get_cell(i_cell)
            for i_point in range(len(current_cell.point_ids)):
                current_point_id: int = current_cell.point_ids[i_point]
                self.cell_neighbour_ids_on_points[current_point_id].append(current_cell_id)
                pass
            pass
        pass

    def setDirichletBoundary(
            self,
            # dirichlet_boundary_bool_function(x, y) -> bool, decide whether the point is a dirichlet boundary point
            dirichlet_boundary_bool_function: 'function', 
            # dirichlet_boundary_value_function(x, y) -> float, get the value of the dirichlet boundary point
            dirichlet_boundary_value_function: 'function'
        ) -> None:
        self.dirichlet_boundary_dict.clear()
        for i_point in range(self.n_points):
            point = self.points[i_point]
            x: float = point[0]
            y: float = point[1]
            if dirichlet_boundary_bool_function(x, y) == True:
                value: float = dirichlet_boundary_value_function(x, y)
                self.dirichlet_boundary_dict[i_point] = value
                pass
            pass
        pass

    def setNeumannBoundary(
            self,
            # neumann_boundary_bool_function(x1, y1, x2, y2) -> bool, decide whether the edge is a neumann boundary edge
            neumann_boundary_bool_function: 'function', 
            # neumann_boundary_value_function(x1, y1, x2, y2) -> float, get the value of the neumann boundary edge
            neumann_boundary_value_function: 'function'
        ) -> None:
        self.neumann_boundary_dict.clear()
        for i_cell in range(self.n_cells):
            current_cell_id: int = i_cell
            current_cell = self.mesh.get_cell(current_cell_id)
            for i in range(len(current_cell.point_ids)):
                point_id_1: int = current_cell.point_ids[i]
                point_id_2: int = current_cell.point_ids[(i + 1) % len(current_cell.point_ids)]
                point_1 = self.points[point_id_1]
                point_2 = self.points[point_id_2]
                x1, y1, _ = point_1
                x2, y2, _ = point_2
                if neumann_boundary_bool_function(x1, y1, x2, y2) == True:
                    value: float = neumann_boundary_value_function(x1, y1, x2, y2)
                    self.neumann_boundary_dict[(point_id_1, point_id_2)] = value
                    pass
                pass
            pass
        pass

    pass