import numpy as np
from bspy import Solid, Boundary, Hyperplane, Viewer
from modeler import Modeler
import extrude

def add_boundaries_to_robot(robot, solid):
    for boundary in solid.boundaries:
        robot.add_boundary(boundary)

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

    modeler.rotate(0, -np.pi / 2)
    modeler.translate(position, dPosition)
    add_boundaries_to_robot(robot, modeler.transform(base))
    modeler.translate((0.0, 0.0, 3.0))
    modeler.rotate(2, hips, dHips)
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.translate((1.0, 0.0, 1.0))
    modeler.rotate(0, shoulder, dShoulder)
    add_boundaries_to_robot(robot, modeler.transform(arm))
    modeler.translate((-0.5, 0.0, 4.0))
    modeler.rotate(0, elbow, dElbow)
    modeler.push()
    modeler.scale((0.5, 1.0, 1.0))
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.pop()
    modeler.translate((-1.5, 0.0, 0.0))
    add_boundaries_to_robot(robot, modeler.transform(arm))
    modeler.translate((0.5, 0.0, 5.5))
    modeler.rotate(2, wrist, dWrist)
    modeler.push()
    modeler.scale((0.5, 0.5, 0.5))
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.pop()
    modeler.translate((0.0, 0.0, 0.5))
    add_boundaries_to_robot(robot, modeler.transform(jaw))
    modeler.push()
    modeler.translate((0.0, bite, 0.5), (0.0, dBite, 0.0))
    add_boundaries_to_robot(robot, modeler.transform(tooth))
    modeler.pop()
    modeler.translate((0.0, -bite, 0.5), (0.0, -dBite, 0.0))
    add_boundaries_to_robot(robot, modeler.transform(tooth))

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
    solid = Solid(3, False)
    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-2.0, 2.0),)*3)

    #modeler.translate((2.0 * t, 0.0, 0.0), (2.0, 0.0, 0.0))
    #modeler.scale((1.0 + 2.0 * t, 1.0, 1.0), (2.0, 0.0, 0.0))
    modeler.rotate(1, t * np.pi / 4, np.pi / 4)
    add_boundaries_to_robot(solid, modeler.transform(base))

    return solid

if __name__ == "__main__":
    viewer = Viewer()
    viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))

    solid = extrude.extrude_time(robot1, 0.0, 1.0, 11)
    hyperplane = Hyperplane.create_axis_aligned(solid.dimension, 3, 0.0)
    for t in np.linspace(0.02, 1.0, 10):
        hyperplane._point = t * hyperplane._normal
        slice = solid.slice(hyperplane)
        viewer.list(slice, fillColor=np.array((0, 1, 0, 1),np.float32))
    viewer.mainloop()
