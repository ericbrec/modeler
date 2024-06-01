import logging
import numpy as np
from bspy import Solid, Boundary, Hyperplane, Viewer
from modeler import Modeler
import utils

def interpolate(t, start, end):
    if t < start[0]:
        return start[1]
    elif t > end[0]:
        return end[1]
    else:
        return (start[1] * (end[0] - t) + end[1] * (t - start[0])) / (end[0] - start[0])

def add_boundaries(solid, modeler, part):
    for boundary in modeler.transform(part).boundaries:
        solid.add_boundary(boundary)

def create_robot(parameters, t, robot = None):
    if robot is None:
        robot = Solid(3, False)
    
    h = 0.0001
    if callable(parameters):
        travel, crossing, height, bite, position = parameters(t)
        dTravel, dCrossing, dHeight, dBite, dPosition = parameters(t + h)
    else:
        travel, crossing, height, bite, position = parameters
        dTravel, dCrossing, dHeight, dBite, dPosition = parameters

    dTravel = (dTravel - travel) / h
    dCrossing = (dCrossing - crossing) / h
    dHeight = (dHeight - height) / h
    dBite = (dBite - bite) / h
    dPosition = (np.array(dPosition) - np.array(position)) / h

    modeler = Modeler()
    nearBase = Hyperplane.create_hypercube(((-2.0, 2.0),(1.0, 1.1), (-2.2, -2.0)))
    farBase = Hyperplane.create_hypercube(((-2.0, 2.0),(1.0, 1.1), (2.0, 2.2)))
    crossBeam = Hyperplane.create_hypercube(((-0.1, 0.1),(1.1, 1.4), (-2.2, 2.2)))
    arm = Hyperplane.create_hypercube(((0.1, 0.3), (-0.8, 0.8), (-0.2, 0.2)))
    jaw = Hyperplane.create_hypercube(((0.09, 0.31), (-0.9, -0.8), (-0.35, 0.35)))
    tooth = Hyperplane.create_hypercube(((0.09, 0.31), (-1.4, -0.9), (-0.02, 0.02)))

    adapter = Hyperplane.create_hypercube(((-0.2, 0.2),(-0.2, 0.2), (-0.25, 0.25)))

    add_boundaries(robot, modeler, nearBase)
    add_boundaries(robot, modeler, farBase)
    modeler.translate((travel, 0.0, 0.0), (dTravel, 0.0, 0.0))
    add_boundaries(robot, modeler, crossBeam)
    modeler.translate((0.0, height, crossing), (0.0, dHeight, dCrossing))
    add_boundaries(robot, modeler, arm)
    add_boundaries(robot, modeler, jaw)
    modeler.push()
    modeler.translate((0.0, 0.0, bite), (0.0, 0.0, dBite))
    add_boundaries(robot, modeler, tooth)
    modeler.pop()
    modeler.translate((0.0, 0.0, -bite), (0.0, 0.0, -dBite))
    add_boundaries(robot, modeler, tooth)

    modeler.reset()
    modeler.translate(position, dPosition)
    add_boundaries(robot, modeler, adapter)

    return robot

def robot_parameters(t):
    if t < 0.5:
        travel = interpolate(t, (0.0, -2.0), (0.5, -0.2))
        crossing = interpolate(t, (0.0, -1.8), (0.5, 1.5))
        height = interpolate(t, (0.0, 1.8), (0.5, 0.8))
        bite = interpolate(t, (0.0, 0.35), (0.5, 0.27))
        position = (0.0, -0.6, 1.5)
    else:
        travel = interpolate(t, (0.5, -0.2), (1.0, 1.5))
        crossing = interpolate(t, (0.5, 1.5), (1.0, -0.6))
        bite = interpolate(t, (0.5, 0.27), (1., 0.27))
        if t < 0.75:
            height = interpolate(t, (0.5, 0.8), (0.75, 1.2))
            position = (
                interpolate(t, (0.5, 0.0), (1.0, 1.7)), 
                interpolate(t, (0.5, -0.6), (0.75, -0.3)), 
                interpolate(t, (0.5, 1.5), (1.0, -0.6))
                )
        else:
            height = interpolate(t, (0.75, 1.2), (1.0, 0.8))
            position = (
                interpolate(t, (0.5, 0.0), (1.0, 1.7)), 
                interpolate(t, (0.75, -0.3), (1.0, -0.6)), 
                interpolate(t, (0.5, 1.5), (1.0, -0.6))
                )
    return (travel, crossing, height, bite, position)

def robot(t):
    return create_robot(robot_parameters, t)

def create_router(parameters, t, router = None):
    if router is None:
        router = Solid(3, False)
    
    h = 0.0001
    if callable(parameters):
        travel = parameters(t)
        dTravel = parameters(t + h)
    else:
        travel = parameters
        dTravel = parameters

    dTravel = (dTravel - travel) / h

    modeler = Modeler()
    base = utils.create_faceted_solid_from_points(((-1.3, -0.8), (-1.3, -0.2), (1.3, -0.6), (1.3, -0.8)))
    base = utils.extrude_path(base, ((0.0, 0.0, -1.6), (0.0, 0.0, 0.4)))
    antenna = utils.create_faceted_solid_from_points(((-0.1, -0.08), (-0.1, 0.08), (1.4, 0.02), (1.4, -0.02)))
    antenna = utils.extrude_path(antenna, ((0.0, 0.0, -0.05), (0.0, 0.0, 0.05)))

    outer = Hyperplane.create_hypercube(((-0.25, 0.25), (-0.3, 0.3)))
    inner = Hyperplane.create_hypercube(((-0.22, 0.22), (-0.27, 0.27)))
    box = utils.extrude_path(outer - inner, ((0.0, 0.0, 0.0), (0.0, 0.0, 0.2)))

    modeler.translate((travel, 0.0, 0.0), (dTravel, 0.0, 0.0))
    add_boundaries(router, modeler, base)
    modeler.push()
    modeler.translate((-1.1, -0.3, 0.45))
    modeler.rotate(2, np.pi / 16)
    add_boundaries(router, modeler, antenna)
    modeler.pop()
    modeler.push()
    modeler.translate((-1.1, -0.3, -1.65))
    modeler.rotate(2, np.pi / 16)
    add_boundaries(router, modeler, antenna)
    modeler.pop()
    modeler.translate((1.7, -0.8, -0.6))
    modeler.rotate(0, -np.pi/ 2)
    add_boundaries(router, modeler, box)

    return router

def router_parameters(t):
    return interpolate(t, (0.0, -2.0), (0.5, 0.0))

def router(t, robot=None):
    return create_router(router_parameters, t, robot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s', datefmt='%H:%M:%S')
    np.set_printoptions(suppress=True)

    option = "draw"

    if option == "build":
        logging.info("Extrude robot")
        extrudedRobot = utils.extrude_time(robot, (0.0, 0.5, 0.75, 1.0))
        logging.info("Extrude router")
        extrudedRouter = utils.extrude_time(router, (0.0, 0.5, 1.0))
        logging.info("Intersect robots")
        intersection = extrudedRobot.intersection(extrudedRouter)
        logging.info("Save intersection")
        Solid.save(r"C:\Users\ericb\OneDrive\Desktop\robots_intersection.json", intersection)

    elif option == "test":
        viewer = Viewer()
        viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))
        rob = robot(0.3)
        viewer.list(rob)
        rou = router(0.3)
        viewer.list(rou)
        cache = {}
        boundary = rob.boundaries[34]
        viewer.list(boundary, "B34")
        slice = rou.slice(boundary.manifold, cache, True)
        viewer.list(Boundary(boundary.manifold, slice), "B34 Slice")
        trimmed = boundary.domain.intersection(slice, cache)
        viewer.list(Boundary(boundary.manifold, trimmed), "B34 Trim")
        inter = rob.intersection(rou)
        viewer.list(inter)
        viewer.mainloop()

    elif option == "draw":
        viewer = Viewer()
        viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))

        logging.info("Render robot animation")
        for t in np.linspace(0.0, 1.0, 11):
            viewer.list(router(t, robot(t)), f"Router {t:.2f}")

        logging.info("Load intersection")
        [intersection] = Solid.load(r"C:\Users\ericb\OneDrive\Desktop\robots_intersection.json")

        logging.info("Slice intersection")
        hyperplane = Hyperplane.create_axis_aligned(4, 3, 0.0)
        for t in np.linspace(0.0, 1.0, 21):
            hyperplane._point = t * hyperplane._normal
            slice = intersection.slice(hyperplane)
            logging.info(f"Intersection {t:.2f}")
            viewer.list(slice, f"Intersection {t:.2f}")
        viewer.mainloop()
