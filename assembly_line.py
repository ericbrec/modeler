import logging
import numpy as np
from bspy import Solid, Boundary, Hyperplane, Viewer
from modeler import Modeler
import extrude

def add_boundaries(solid, modeler, part):
    for boundary in modeler.transform(part).boundaries:
        solid.add_boundary(boundary)

def create_robot(parameters, t, robot = None):
    if robot is None:
        robot = Solid(3, False)
    
    h = 0.0001
    if callable(parameters):
        travel, crossing, height, bite = parameters(t)
        dTravel, dCrossing, dHeight, dBite = parameters(t + h)
    else:
        travel, crossing, height, bite = parameters
        dTravel, dCrossing, dHeight, dBite = parameters

    dTravel = (dTravel - travel) / h
    dCrossing = (dCrossing - crossing) / h
    dHeight = (dHeight - height) / h
    dBite = (dBite - bite) / h

    modeler = Modeler()
    nearBase = Hyperplane.create_hypercube(((-2.0, 2.0),(1.0, 1.1), (-2.2, -2.0)))
    farBase = Hyperplane.create_hypercube(((-2.0, 2.0),(1.0, 1.1), (2.0, 2.2)))
    crossBeam = Hyperplane.create_hypercube(((-0.1, 0.1),(1.1, 1.4), (-2.2, 2.2)))
    arm = Hyperplane.create_hypercube(((0.1, 0.3), (-0.8, 1.2), (-0.2, 0.2)))
    jaw = Hyperplane.create_hypercube(((0.09, 0.31), (-0.9, -0.8), (-0.35, 0.35)))
    tooth = Hyperplane.create_hypercube(((0.09, 0.31), (-1.4, -0.9), (-0.02, 0.02)))

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

    return robot

def robot_parameters(t):
    travel = (1 - t) * -2.0 + t * 2.0
    crossing = (1 - t) * -1.8 + t * 1.8
    height = (1 - t) * 0.0  + t * 1.8
    bite = 0.35 if t < 0.5 else (2 - 2 * t) * 0.35 + (2 * t - 1) * 0.02
    return (travel, crossing, height, bite)

def robot(t):
    return create_robot(robot_parameters, t)

def create_router(parameters, t, router = None):
    if router is None:
        router = Solid(3, False)
    
    h = 0.0001
    if callable(parameters):
        position = parameters(t)
        dPosition = parameters(t + h)
    else:
        position = parameters
        dPosition = parameters

    dPosition = (np.array(dPosition) - np.array(position)) / h

    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-1.3, 1.3), (-0.8, -0.2), (-1.6, 0.4)))
    antenna = Hyperplane.create_hypercube(((-0.1, 1.4),(-0.08, 0.08), (-0.05, 0.05)))
    adapter = Hyperplane.create_hypercube(((-0.2, 0.2),(-0.8, -0.4), (0.75, 1.25)))

    add_boundaries(router, modeler, base)
    modeler.push()
    modeler.translate((-1.2, -0.2, 0.42))
    modeler.rotate(2, np.pi / 4)
    add_boundaries(router, modeler, antenna)
    modeler.pop()
    modeler.push()
    modeler.translate((-1.2, -0.2, -1.62))
    modeler.rotate(2, 3 * np.pi / 4)
    add_boundaries(router, modeler, antenna)
    modeler.pop()
    modeler.translate(position, dPosition)
    add_boundaries(router, modeler, adapter)

    return router

def router_parameters(t):
    position = (0.0, 0.0, 0.0)
    return position

def router(t):
    return create_router(router_parameters, t)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s', datefmt='%H:%M:%S')
    np.set_printoptions(suppress=True)

    option = "draw"
    if option == "build":
        logging.info("Extrude robot1")
        extruded1 = extrude.extrude_time(robot, 0.6, 1.0, 3)
        logging.info("Extrude robot2")
        extruded2 = extrude.extrude_time(robot2, 0.6, 1.0, 3)
        logging.info("Intersect robots")
        intersection = extruded1.intersection(extruded2)
        logging.info("Save intersection")
        Solid.save(r"C:\Users\ericb\OneDrive\Desktop\robots_intersection.json", intersection)

    elif option == "test":
        solid = robot(0.8)
        viewer = Viewer()
        viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))
        logging.info("Extrude robot")
        extruded1 = extrude.extrude_time(robot, 0.0, 1.0, 2)
        logging.info("Slice intersection")
        Hyperplane.maxAlignment = 0.9999
        hyperplane = Hyperplane.create_axis_aligned(4, 3, 0.0)
        for t in np.linspace(0.02, 0.98, 11):
            hyperplane._point = t * hyperplane._normal
            slice = extruded1.slice(hyperplane)
            logging.info(f"Slice1 {t:.1f}")
            viewer.list(slice, f"Slice1 {t:.1f}")
        viewer.mainloop()

    elif option == "draw":
        viewer = Viewer()
        viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))

        logging.info("Render robot animation")
        t = 0.5
        viewer.list(robot(t), f"Robot {t:.1f}")
        viewer.list(router(t), f"Router {t:.1f}")
        viewer.mainloop()
