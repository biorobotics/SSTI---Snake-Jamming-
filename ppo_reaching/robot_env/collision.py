"""MuJoCo convex contact, planar bounds and link clearance."""

import mujoco as mj
import numpy as np


PLANE = np.array([[0., 0., -1.], [1., 0., 0.]])
LOW = np.array([0., -.5])
HIGH = np.array([1., .5])
MODULES = 12
OBSTACLE_MASK = 1 << MODULES


class Geometry:
    def __init__(self, model, data, robot, modules, obstacles):
        self.m, self.d = model, data
        self.robot, self.modules = np.array(robot), np.array(modules)
        self.obstacles = np.asarray(obstacles, dtype=int)
        self.obstacle_ids = set(obstacles)
        self.robot_radii = model.geom_rbound[self.robot]
        self.obstacle_radii = model.geom_rbound[self.obstacles]
        self.mesh_vertices = {}
        for geom in robot:
            if model.geom_type[geom] == mj.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[geom]
                begin = model.mesh_vertadr[mesh]
                self.mesh_vertices[geom] = model.mesh_vert[begin:begin + model.mesh_vertnum[mesh]]
            elif model.geom_type[geom] != mj.mjtGeom.mjGEOM_CYLINDER:
                raise ValueError("Unsupported robot geometry")

    def contact(self, clearance):
        contacts = self.contacts(clearance)
        return next(iter(contacts), None)

    def contacts(self, clearance):
        contacts = {}
        for index in range(self.d.ncon):
            contact = self.d.contact[index]
            if contact.dist <= clearance + 1e-10:
                is_obstacle = contact.geom1 in self.obstacle_ids or contact.geom2 in self.obstacle_ids
                status = "collision" if is_obstacle else "self_collision"
                if status not in contacts:
                    contacts[status] = (PLANE @ contact.pos).tolist()
        return contacts

    def in_box(self):
        centers = self.d.geom_xpos[self.robot] @ PLANE.T
        sphere_bound = np.minimum(centers - LOW, HIGH - centers).min(axis=1) - self.robot_radii
        for index in np.flatnonzero(sphere_bound < 0.):
            geom = self.robot[index]
            projection = PLANE @ self.d.geom_xmat[geom].reshape(3, 3)
            if geom in self.mesh_vertices:
                projected = self.mesh_vertices[geom] @ projection.T + centers[index]
                if np.any(projected < LOW) or np.any(projected > HIGH):
                    return False
            else:
                radius, half_length = self.m.geom_size[geom, :2]
                axis = np.clip(np.abs(projection[:, 2]), 0., 1.)
                extent = radius * np.sqrt(1. - axis**2) + half_length * axis
                if np.any(centers[index] - extent < LOW) or np.any(centers[index] + extent > HIGH):
                    return False
        return True

    def link_clearance(self, cap):
        clearance = np.full(MODULES, cap)
        if not len(self.obstacles):
            return clearance
        delta = self.d.geom_xpos[self.robot, None, :] - self.d.geom_xpos[self.obstacles]
        lower = np.linalg.norm(delta, axis=2) - self.robot_radii[:, None] - self.obstacle_radii
        for i, j in zip(*np.where(lower < cap)):
            module = self.modules[i]
            if lower[i, j] < clearance[module]:
                distance = mj.mj_geomDistance(self.m, self.d, self.robot[i], self.obstacles[j],
                                              clearance[module], None)
                clearance[module] = min(clearance[module], max(0., distance))
        return clearance


def make_obstacle(world, index, box, height, margin):
    import xml.etree.ElementTree as ET

    longitudinal, transverse, length, width, theta = box
    ET.SubElement(world, "geom", name=f"rl_obstacle_{index}", type="box",
                  pos=f"{transverse} {height} {-longitudinal}",
                  size=f"{width / 2} .11 {length / 2}", axisangle=f"0 -1 0 {theta}",
                  contype=str(OBSTACLE_MASK), conaffinity="0", margin=str(margin),
                  rgba=".32 .44 .60 1")
