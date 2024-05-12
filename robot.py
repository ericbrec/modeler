import numpy as np
from bspy import Solid, Hyperplane, Viewer
from modeler import Modeler

def add_boundaries_to_robot(robot, solid):
    for boundary in solid.boundaries:
        robot.add_boundary(boundary)

def create_robot(position, hips, shoulder, elbow, wrist, bite, robot = None):
    if robot is None:
        robot = Solid(3, False)
    
    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-2.0, 2.0),)*3)
    pivot = Hyperplane.create_hypercube(((-1.0, 1.0),)*3)
    arm = Hyperplane.create_hypercube(((0.0, 1.0), (-1.0, 1.0), (-1.0, 5.0)))
    jaw = Hyperplane.create_hypercube(((-0.5, 0.5), (-1.5, 1.5), (0., 0.5)))
    tooth = Hyperplane.create_hypercube(((-0.5, 0.5), (-0.2, 0.2), (0.0, 1.5)))

    modeler.rotate(0, -np.pi / 2)
    modeler.translate(position)
    add_boundaries_to_robot(robot, modeler.transform(base))
    modeler.translate(0.0, 0.0, 3.0)
    modeler.rotate(2, hips)
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.translate(1.0, 0.0, 1.0)
    modeler.rotate(0, shoulder)
    add_boundaries_to_robot(robot, modeler.transform(arm))
    modeler.translate(-0.5, 0.0, 4.0)
    modeler.rotate(0, elbow)
    modeler.push()
    modeler.scale(0.5, 1.0, 1.0)
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.pop()
    modeler.translate(-1.5, 0.0, 0.0)
    add_boundaries_to_robot(robot, modeler.transform(arm))
    modeler.translate(0.5, 0.0, 5.5)
    modeler.rotate(2, wrist)
    modeler.push()
    modeler.scale(0.5, 0.5, 0.5)
    add_boundaries_to_robot(robot, modeler.transform(pivot))
    modeler.pop()
    modeler.translate(0.0, 0.0, 0.5)
    add_boundaries_to_robot(robot, modeler.transform(jaw))
    modeler.push()
    modeler.translate(0.0, bite, 0.5)
    add_boundaries_to_robot(robot, modeler.transform(tooth))
    modeler.pop()
    modeler.translate(0.0, -bite, 0.5)
    add_boundaries_to_robot(robot, modeler.transform(tooth))

    return robot

def robot1_parameters(t):
    position = (4.0, 0.0, 0.0)
    hips = (1 - t) * 0 * np.pi / 2  + t * np.pi / -4
    shoulder = (1 - t) * np.pi / -4  + t * np.pi / 4
    elbow = (1 - t) * np.pi / 2 + t * 2 * np.pi / 4
    wrist = (1 - t) * 0 + t * np.pi / 2
    bite = 1.0 if t < 0.5 else (2 - 2 * t) * 1.0 + (2 * t - 1) * 0.5
    return (position, hips, shoulder, elbow, wrist, bite)

def robot2_parameters(t):
    position = (-4.0, 0.0, 0.0)
    hips = (1 - t) * 2 * np.pi / -2  + t * 3 * np.pi / -4
    shoulder = (1 - t) * np.pi / 2  + t * 2 * np.pi / -4
    elbow = (1 - t) * np.pi / -2 + t * 1 * np.pi / -4
    wrist = (1 - t) * 0 + t * np.pi / -2
    bite = 1.0 if t < 0.5 else (2 - 2 * t) * 1.0 + (2 * t - 1) * 0.5
    return (position, hips, shoulder, elbow, wrist, bite)

if __name__ == "__main__":
    viewer = Viewer()
    for t in np.linspace(0.0, 1.0, 11):
        robot = None
        robot = create_robot(*robot1_parameters(t))
        robot = create_robot(*robot2_parameters(t), robot)
        viewer.list(robot)
    viewer.mainloop()
