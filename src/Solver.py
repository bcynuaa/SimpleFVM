'''
 # @ author: bcy | htc
 # @ date: 2024-05-20 17:31:44
 # @ license: MIT
 # @ description:  
 '''

import os
import numpy as np
import tqdm

from MyMesh import MyMesh

class Solver:

    # solve such equation:
    # ut = alpha * (uxx + uyy)

    def __init__(self) -> None:
        self.alpha: float = 1.0
        self.dt: float = 0.01
        self.max_iteration: int = 100
        pass

    def combineMesh(self, mesh: MyMesh) -> None:
        self.mesh: MyMesh = mesh
        self.u_on_points: np.ndarray = np.zeros(self.mesh.n_points)
        self.u_on_cells: np.ndarray = np.zeros(self.mesh.n_cells)
        self.nabla_u_on_points: np.ndarray = np.zeros((self.mesh.n_points, 2))
        self.nabla_u_on_cells: np.ndarray = np.zeros((self.mesh.n_cells, 2))
        self.flux_on_cells: np.ndarray = np.zeros(self.mesh.n_cells)
        pass

    def setAlpha(self, alpha: float) -> None:
        self.alpha = alpha
        pass

    def setDt(self, dt: float) -> None:
        self.dt = dt
        pass

    def setMaxIteration(self, max_iteration: int) -> None:
        self.max_iteration = max_iteration
        pass

    def cellDataToPointData(self) -> None:
        for i_point in range(self.mesh.n_points):
            cell_neighbour_ids = self.mesh.cell_neighbour_ids_on_points[i_point]
            cell_neighbour_values: np.ndarray = self.u_on_cells[cell_neighbour_ids]
            cell_neighbour_sizes: np.ndarray = self.mesh.area[cell_neighbour_ids]
            self.u_on_points[i_point] = np.dot(cell_neighbour_values, cell_neighbour_sizes) / np.sum(cell_neighbour_sizes)
            pass
        pass

    def pointDataToCellData(self) -> None:
        for i_cell in range(self.mesh.n_cells):
            current_cell = self.mesh.mesh.get_cell(i_cell)
            value_sum: float = 0.0
            value_count: int = 0
            for i_point in range(len(current_cell.point_ids)):
                current_point_id = current_cell.point_ids[i_point]
                value_sum += self.u_on_points[current_point_id]
                value_count += 1
                pass
            self.u_on_cells[i_cell] = value_sum / value_count
            pass
        pass

    def applyDirichletBoundary(self) -> None:
        point_id = np.array(list(self.mesh.dirichlet_boundary_dict.keys()), dtype=int)
        self.u_on_points[point_id] = np.array(list(self.mesh.dirichlet_boundary_dict.values()))
        pass

    def computeNblaUOnCells(self) -> None:
        for i_cell in range(self.mesh.n_cells):
            current_cell = self.mesh.mesh.get_cell(i_cell)
            nabla_u: np.ndarray = np.zeros(2)
            for i in range(len(current_cell.point_ids)):
                j = (i + 1) % len(current_cell.point_ids)
                point_id_1 = current_cell.point_ids[i]
                point_id_2 = current_cell.point_ids[j]
                dx: float = self.mesh.points[point_id_2][0] - self.mesh.points[point_id_1][0]
                dy: float = self.mesh.points[point_id_2][1] - self.mesh.points[point_id_1][1]
                mean_u: float = 0.5 * (self.u_on_points[point_id_1] + self.u_on_points[point_id_2])
                nabla_u += mean_u * np.array([dy, -dx])
                pass
            self.nabla_u_on_cells[i_cell] = nabla_u / self.mesh.area[i_cell]
            pass
        pass

    def cellGradientToPointGradient(self) -> None:
        for i_point in range(self.mesh.n_points):
            cell_neighbour_ids = self.mesh.cell_neighbour_ids_on_points[i_point]
            cell_neighbour_gradients: np.ndarray = self.nabla_u_on_cells[cell_neighbour_ids, :]
            cell_neighbour_sizes: np.ndarray = self.mesh.area[cell_neighbour_ids]
            sum_weighted_gradients: np.ndarray = np.dot(cell_neighbour_gradients.T, cell_neighbour_sizes)
            self.nabla_u_on_points[i_point] = sum_weighted_gradients / np.sum(cell_neighbour_sizes)
            pass
        pass

    def resetFlux(self) -> None:
        self.flux_on_cells.fill(0.0)
        pass

    def addFlux(self) -> None:
        for i_cell in range(self.mesh.n_cells):
            current_cell = self.mesh.mesh.get_cell(i_cell)
            for i in range(len(current_cell.point_ids)):
                j = (i + 1) % len(current_cell.point_ids)
                point_id_1 = current_cell.point_ids[i]
                point_id_2 = current_cell.point_ids[j]
                dx: float = self.mesh.points[point_id_2][0] - self.mesh.points[point_id_1][0]
                dy: float = self.mesh.points[point_id_2][1] - self.mesh.points[point_id_1][1]
                mean_nabla_u: np.ndarray = 0.5 * (self.nabla_u_on_points[point_id_1] + self.nabla_u_on_points[point_id_2])
                if (point_id_1, point_id_2) in self.mesh.neumann_boundary_dict:
                    neumann_value: float = self.mesh.neumann_boundary_dict[(point_id_1, point_id_2)]
                    edge_length: float = np.sqrt(dx * dx + dy * dy)
                    flux: float = neumann_value * edge_length * self.alpha / self.mesh.area[i_cell]
                    pass
                else:
                    flux: float = np.dot(mean_nabla_u, np.array([dy, -dx])) * self.alpha / self.mesh.area[i_cell]
                    pass
                self.flux_on_cells[i_cell] += flux
                pass
            pass
        pass

    def eulerForward(self) -> None:
        self.u_on_cells += self.dt * self.flux_on_cells
        pass

    def initialize(self) -> None:
        self.applyDirichletBoundary()
        self.pointDataToCellData()
        pass

    def eachStep(self) -> None:
        self.cellDataToPointData()
        self.applyDirichletBoundary()
        self.computeNblaUOnCells()
        self.cellGradientToPointGradient()
        self.resetFlux()
        self.addFlux()
        self.eulerForward()
        pass

    def combinePointDataToMesh(self) -> None:
        self.mesh.mesh.point_data["u"] = self.u_on_points
        pass

    def combineCellDataToMesh(self) -> None:
        self.mesh.mesh.cell_data["u"] = self.u_on_cells
        pass

    def saveResult(self, step: int, save_dir: str) -> None:
        self.combinePointDataToMesh()
        self.mesh.mesh.save(
            os.path.join(save_dir, f"result_{step}.vtk")
        )
        pass

    def solve(self, save_dir: str) -> None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            pass
        self.initialize()
        for i in tqdm.tqdm(range(self.max_iteration)):
            self.eachStep()
            self.saveResult(i, save_dir)
            pass
        pass

    pass