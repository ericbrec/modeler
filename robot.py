import logging
import numpy as np
from bspy import Solid, Boundary, Hyperplane, Viewer
from modeler import Modeler
import extrude

def create_robot(parameters, t, robot = None):
    if robot is None:
        robot = Solid(3, False)
    
    h = 0.0001
    if callable(parameters):
        position, hips, shoulder, elbow, wrist, bite = parameters(t)
        dPosition, dHips, dShoulder, dElbow, dWrist, dBite = parameters(t + h)
    else:
        position, hips, shoulder, elbow, wrist, bite = parameters
        dPosition, dHips, dShoulder, dElbow, dWrist, dBite = parameters

    dPosition = (np.array(dPosition) - np.array(position)) / h
    dHips = (dHips - hips) / h
    dShoulder = (dShoulder - shoulder) / h
    dElbow = (dElbow - elbow) / h
    dWrist = (dWrist - wrist) / h
    dBite = (dBite - bite) / h

    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-2.0, 2.0),)*3)
    pivot = Hyperplane.create_hypercube(((-1.0, 1.0),)*3)
    arm = Hyperplane.create_hypercube(((0.0, 1.0), (-1.0, 1.0), (-1.0, 5.0)))
    jaw = Hyperplane.create_hypercube(((-0.5, 0.5), (-1.5, 1.5), (0., 0.5)))
    tooth = Hyperplane.create_hypercube(((-0.5, 0.5), (-0.2, 0.2), (0.0, 1.5)))

    def add_boundaries(solid):
        sld = modeler.transform(solid)
        for boundary in sld.boundaries:
            robot.add_boundary(boundary)

    modeler.rotate(0, -np.pi / 2)
    modeler.translate(position, dPosition)
    add_boundaries(base)
    modeler.translate((0.0, 0.0, 3.0))
    modeler.rotate(2, hips, dHips)
    add_boundaries(pivot)
    modeler.translate((1.0, 0.0, 1.0))
    modeler.rotate(0, shoulder, dShoulder)
    add_boundaries(arm)
    modeler.translate((-0.5, 0.0, 4.0))
    modeler.rotate(0, elbow, dElbow)
    modeler.push()
    modeler.scale((0.5, 1.0, 1.0))
    add_boundaries(pivot)
    modeler.pop()
    modeler.translate((-1.5, 0.0, 0.0))
    add_boundaries(arm)
    modeler.translate((0.5, 0.0, 5.5))
    modeler.rotate(2, wrist, dWrist)
    modeler.push()
    modeler.scale((0.5, 0.5, 0.5))
    add_boundaries(pivot)
    modeler.pop()
    modeler.translate((0.0, 0.0, 0.5))
    add_boundaries(jaw)
    modeler.push()
    modeler.translate((0.0, bite, 0.5), (0.0, dBite, 0.0))
    add_boundaries(tooth)
    modeler.pop()
    modeler.translate((0.0, -bite, 0.5), (0.0, -dBite, 0.0))
    add_boundaries(tooth)

    return robot

def robot1_parameters(t):
    position = (4.0, 0.0, 0.0)
    hips = (1 - t) * np.pi / 5  + t * np.pi / -4
    shoulder = (1 - t) * np.pi / -4  + t * np.pi / 4
    elbow = (1 - t) * np.pi / 2 + t * 2 * np.pi / 4
    wrist = (1 - t) * 0 + t * np.pi / 2
    bite = 1.0 if t < 0.5 else (2 - 2 * t) * 1.0 + (2 * t - 1) * 0.5
    return (position, hips, shoulder, elbow, wrist, bite)

def robot1(t):
    return create_robot(robot1_parameters, t)

def robot2_parameters(t):
    position = (-4.0, 0.0, 0.0)
    hips = (1 - t) * 2 * np.pi / -2  + t * 3 * np.pi / -4
    shoulder = (1 - t) * np.pi / 2  + t * 2 * np.pi / -4
    elbow = (1 - t) * np.pi / -2 + t * 1 * np.pi / -4
    wrist = (1 - t) * 0 + t * np.pi / -2
    bite = 1.0 if t < 0.5 else (2 - 2 * t) * 1.0 + (2 * t - 1) * 0.5
    return (position, hips, shoulder, elbow, wrist, bite)

def robot2(t):
    return create_robot(robot2_parameters, t)

def bar(t):
    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-2.0, 2.0),)*3)
    #modeler.translate((2.0 * t, 0.0, 0.0), (2.0, 0.0, 0.0))
    #modeler.scale((1.0 + 2.0 * t, 1.0, 1.0), (2.0, 0.0, 0.0))
    modeler.rotate(1, t * np.pi / 4, np.pi / 4)
    return modeler.transform(base)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(levelname)s:%(module)s:%(lineno)d:%(message)s', datefmt='%H:%M:%S')
    np.set_printoptions(suppress=True)
    viewer = Viewer()
    viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))

    logging.info("Render robot animation")
    for t in np.linspace(0.02, 0.98, 11):
        robot = create_robot(robot1_parameters, t)
        robot = create_robot(robot2_parameters, t, robot)
        viewer.list(robot)

    logging.info("Extrude robot1")
    extruded1 = extrude.extrude_time(robot1, 0.0, 1.0, 8)
    logging.info("Extrude robot2")
    extruded2 = extrude.extrude_time(robot2, 0.0, 1.0, 8)
    logging.info("Intersect robots")
    intersection = extruded1.intersection(extruded2)
    logging.info("Slice intersection")
    Hyperplane.maxAlignment = 0.9999
    hyperplane = Hyperplane.create_axis_aligned(4, 3, 0.0)
    for t in np.linspace(0.02, 0.98, 11):
        hyperplane._point = t * hyperplane._normal
        slice = intersection.slice(hyperplane)
        logging.info(f"Slice {t}")
        viewer.list(slice, fillColor=np.array((0, 1, 0, 1),np.float32))
    viewer.mainloop()
