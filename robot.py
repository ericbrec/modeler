import numpy as np
from bspy import Solid, Hyperplane, Viewer
from modeler import Modeler

def add_boundaries_to_robot(robot, solid):
    for boundary in solid.boundaries:
        robot.add_boundary(boundary)

def create_robot(hips, shoulder, elbow, wrist, bite):
    robot = Solid(3, False)
    modeler = Modeler()
    base = Hyperplane.create_hypercube(((-2.0, 2.0),)*3)
    pivot = Hyperplane.create_hypercube(((-1.0, 1.0),)*3)
    arm = Hyperplane.create_hypercube(((0.0, 1.0), (-1.0, 1.0), (-1.0, 5.0)))
    jaw = Hyperplane.create_hypercube(((-0.5, 0.5), (-1.5, 1.5), (0., 0.5)))
    tooth = Hyperplane.create_hypercube(((-0.5, 0.5), (-0.2, 0.2), (0.0, 1.5)))

    modeler.rotate(0, -np.pi / 2)
    add_boundaries_to_robot(robot, base)
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

if __name__ == "__main__":
    viewer = Viewer()
    viewer.draw(create_robot(np.pi / 4, -np.pi / 4, 3 * np.pi / 4, np.pi / 2, 1.0))
    viewer.list(create_robot(np.pi / 4, -np.pi / 4, 3 * np.pi / 4, np.pi / 2, 0.5))
    viewer.list(create_robot(np.pi / 4, -np.pi / 7, 3 * np.pi / 4, np.pi / 3, 0.7))
    viewer.mainloop()
