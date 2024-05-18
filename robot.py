import numpy as np
from bspy import Solid, Boundary, Hyperplane, Viewer
from modeler import Modeler

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

def robot2_parameters(t):
    position = (-4.0, 0.0, 0.0)
    hips = (1 - t) * 2 * np.pi / -2  + t * 3 * np.pi / -4
    shoulder = (1 - t) * np.pi / 2  + t * 2 * np.pi / -4
    elbow = (1 - t) * np.pi / -2 + t * 1 * np.pi / -4
    wrist = (1 - t) * 0 + t * np.pi / -2
    bite = 1.0 if t < 0.5 else (2 - 2 * t) * 1.0 + (2 * t - 1) * 0.5
    return (position, hips, shoulder, elbow, wrist, bite)

def extrude_solid(solid, path):
    assert len(path) > 1
    assert solid.dimension+1 == len(path[0])
    
    extrusion = Solid(solid.dimension+1, False)

    # Extrude boundaries along the path
    point = None
    for nextPoint in path:
        nextPoint = np.atleast_1d(nextPoint)
        if point is None:
            point = nextPoint
            continue
        tangent = nextPoint - point
        extent = tangent[solid.dimension]
        tangent = tangent / extent
        # Extrude each boundary
        for boundary in solid.boundaries:
            # Construct a normal orthogonal to both the boundary tangent space and the path tangent
            extruded_normal = np.full((extrusion.dimension), 0.0)
            extruded_normal[0:solid.dimension] = boundary.manifold._normal[:]
            extruded_normal[solid.dimension] = -np.dot(boundary.manifold._normal, tangent[0:solid.dimension])
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct a point that adds the boundary point to the path point
            extruded_point = np.full((extrusion.dimension), 0.0)
            extruded_point[0:solid.dimension] = boundary.manifold._point[:]
            extruded_point += point
            # Combine the boundary tangent space and the path tangent
            extruded_tangentSpace = np.full((extrusion.dimension, solid.dimension), 0.0)
            if solid.dimension > 1:
                extruded_tangentSpace[0:solid.dimension, 0:solid.dimension-1] = boundary.manifold._tangentSpace[:,:]
            extruded_tangentSpace[:, solid.dimension-1] = tangent[:]
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            # Construct a domain for the extruded boundary
            if boundary.domain.dimension > 0:
                # Extrude the boundary's domain to include path domain
                domainPath = []
                domainPoint = np.full((solid.dimension), 0.0)
                domainPath.append(domainPoint)
                domainPoint = np.full((solid.dimension), 0.0)
                domainPoint[solid.dimension-1] = extent
                domainPath.append(domainPoint)
                extrudedDomain = extrude_solid(boundary.domain, domainPath)
            else:
                extrudedDomain = Solid(solid.dimension, False)
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(1.0, extent), Solid(0, True)))
            # Add extruded boundary
            extrusion.boundaries.append(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next point
        point = nextPoint

    # Add end cap boundaries
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, True)
    extrudedHyperplane = extrudedHyperplane.translate(path[0])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, False)
    extrudedHyperplane = extrudedHyperplane.translate(path[-1])
    extrusion.boundaries.append(Boundary(extrudedHyperplane, solid))

    return extrusion

"""
def add_time_dimension(solidFunction, t1, t2, samples):
    assert(samples >= 2)
    solid = solidFunction(t1)
    solidPlusTime = Solid(solid.dimension + 1, solid.containsInfinity)

    # Start with t1 solid cap
    cap = Hyperplane.create_axis_aligned(solidPlusTime.dimension, solidPlusTime.dimension - 1, -t1, True)
    solidPlusTime.boundaries.append(Boundary(extrudedHyperplane, solid))

    # Interpolate boundaries along the path
    previousSolid = solid
    samples -= 1 # Simplify arithmetic
    for sample in range(1, samples + 1):
        t = (t1 * (samples - sample) + t2 * sample) / samples
        solid = solidFunction(t)
        tangent = nextPoint - point
        extent = tangent[solid.dimension]
        tangent = tangent / extent
        # Interpolate each boundary.
        for boundary in solid.boundaries:
            # Construct a normal orthogonal to both the boundary tangent space and the path tangent
            extruded_normal = np.full((solidPlusTime.dimension), 0.0)
            extruded_normal[0:solid.dimension] = boundary.manifold._normal[:]
            extruded_normal[solid.dimension] = -np.dot(boundary.manifold._normal, tangent[0:solid.dimension])
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct a point that adds the boundary point to the path point
            extruded_point = np.full((solidPlusTime.dimension), 0.0)
            extruded_point[0:solid.dimension] = boundary.manifold._point[:]
            extruded_point += point
            # Combine the boundary tangent space and the path tangent
            extruded_tangentSpace = np.full((solidPlusTime.dimension, solid.dimension), 0.0)
            if solid.dimension > 1:
                extruded_tangentSpace[0:solid.dimension, 0:solid.dimension-1] = boundary.manifold._tangentSpace[:,:]
            extruded_tangentSpace[:, solid.dimension-1] = tangent[:]
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            # Construct a domain for the extruded boundary
            if boundary.domain.dimension > 0:
                # Extrude the boundary's domain to include path domain
                domainPath = []
                domainPoint = np.full((solid.dimension), 0.0)
                domainPath.append(domainPoint)
                domainPoint = np.full((solid.dimension), 0.0)
                domainPoint[solid.dimension-1] = extent
                domainPath.append(domainPoint)
                extrudedDomain = extrude_solid(boundary.domain, domainPath)
            else:
                extrudedDomain = Solid(solid.dimension, False)
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.boundaries.append(Boundary(hyperplane_1D(1.0, extent), Solid(0, True)))
            # Add extruded boundary
            solidPlusTime.boundaries.append(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next point
        point = nextPoint

    # Add end cap boundaries
    extrudedHyperplane = Hyperplane.create_axis_aligned(solidPlusTime.dimension, solid.dimension, 0.0, True)
    extrudedHyperplane = extrudedHyperplane.translate(path[0])
    solidPlusTime.boundaries.append(Boundary(extrudedHyperplane, solid))
    extrudedHyperplane = Hyperplane.create_axis_aligned(solidPlusTime.dimension, solid.dimension, 0.0, False)
    extrudedHyperplane = extrudedHyperplane.translate(path[-1])
    solidPlusTime.boundaries.append(Boundary(extrudedHyperplane, solid))

    return solidPlusTime
"""

if __name__ == "__main__":
    viewer = Viewer()
    viewer.set_background_color(np.array((1, 1, 1, 1),np.float32))
    for t in np.linspace(0.0, 1.0, 11):
        robot = None
        robot = create_robot(robot1_parameters, t)
        robot = create_robot(robot2_parameters, t, robot)
        viewer.list(robot, fillColor=np.array((0, 1, 0, 1),np.float32))
    viewer.mainloop()
