import numpy as np
from bspy import Solid, Boundary, Hyperplane

def hyperplane_1D(normal, offset):
    assert np.isscalar(normal) or len(normal) == 1
    normalizedNormal = np.atleast_1d(normal)
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    return Hyperplane(normalizedNormal, offset * normalizedNormal, 0.0)

def extrude_path(solid, path):
    assert len(path) > 1
    assert solid.dimension+1 == len(path[0])
    
    extrusion = Solid(solid.dimension + 1, False)

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
            extruded_normal = np.zeros(extrusion.dimension)
            extruded_normal[:solid.dimension] = boundary.manifold._normal[:]
            extruded_normal[solid.dimension] = -np.dot(boundary.manifold._normal, tangent[:solid.dimension])
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct a point that adds the boundary point to the path point
            extruded_point = np.zeros(extrusion.dimension)
            extruded_point[:solid.dimension] = boundary.manifold._point[:]
            extruded_point += point
            # Combine the boundary tangent space and the path tangent
            extruded_tangentSpace = np.zeros((extrusion.dimension, solid.dimension))
            if solid.dimension > 1:
                extruded_tangentSpace[:solid.dimension, :solid.dimension-1] = boundary.manifold._tangentSpace[:,:]
            extruded_tangentSpace[:, solid.dimension-1] = tangent[:]
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            # Construct a domain for the extruded boundary
            if boundary.domain.dimension > 0:
                # Extrude the boundary's domain to include path domain
                domainPath = []
                domainPoint = np.zeros(solid.dimension)
                domainPath.append(domainPoint)
                domainPoint = np.zeros(solid.dimension)
                domainPoint[solid.dimension-1] = extent
                domainPath.append(domainPoint)
                extrudedDomain = extrude_path(boundary.domain, domainPath)
            else:
                extrudedDomain = Solid(solid.dimension, False)
                extrudedDomain.add_boundary(Boundary(hyperplane_1D(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.add_boundary(Boundary(hyperplane_1D(1.0, extent), Solid(0, True)))
            # Add extruded boundary
            extrusion.add_boundary(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next point
        point = nextPoint

    # Add end cap boundaries
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, True)
    extrudedHyperplane = extrudedHyperplane.translate(path[0])
    extrusion.add_boundary(Boundary(extrudedHyperplane, solid))
    extrudedHyperplane = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, 0.0, False)
    extrudedHyperplane = extrudedHyperplane.translate(path[-1])
    extrusion.add_boundary(Boundary(extrudedHyperplane, solid))

    return extrusion

def extrude_time(solidFunction, t1, t2, samples):
    assert(samples >= 2)

    # Start with t1 solid cap.
    solid = solidFunction(t1)
    extrusion = Solid(solid.dimension + 1, solid.containsInfinity)
    cap = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, -t1, True)
    extrusion.add_boundary(Boundary(cap, solid))

    # Interpolate boundaries along time dimension.
    samples -= 1 # Simplify arithmetic
    t = t1
    dT = 1.0 / samples
    for sample in range(1, samples + 1):
        # Interpolate each boundary.
        for boundary in solid.boundaries:
            manifold = boundary.manifold
            # Construct the normal
            extruded_normal = np.zeros(extrusion.dimension)
            extruded_normal[:solid.dimension] = manifold._normal[:]
            extruded_normal[solid.dimension] = -np.dot(manifold._normal, manifold._dP)
            extruded_normal = extruded_normal / np.linalg.norm(extruded_normal)
            # Construct the point
            extruded_point = np.zeros(extrusion.dimension)
            extruded_point[:solid.dimension] = manifold._point[:]
            extruded_point[solid.dimension] = t
            # Construct the tangent space
            extruded_tangentSpace = np.zeros((extrusion.dimension, solid.dimension))
            extruded_tangentSpace[:solid.dimension, :solid.dimension-1] = manifold._tangentSpace[:,:]
            extruded_tangentSpace[:solid.dimension, solid.dimension-1] = manifold._dP
            extruded_tangentSpace[solid.dimension, solid.dimension-1] = 1.0
            # Construct the domain (extrude existing domain to add time)
            domainPath = []
            domainPoint = np.zeros(solid.dimension)
            domainPath.append(domainPoint)
            domainPoint = np.zeros(solid.dimension)
            domainPoint[solid.dimension-1] = dT
            domainPath.append(domainPoint)
            extrudedDomain = extrude_path(boundary.domain, domainPath)
            # Add extruded boundary
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            extrusion.add_boundary(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Move onto the next time
        t = (t1 * (samples - sample) + t2 * sample) / samples
        solid = solidFunction(t)

    # End with t2 solid cap.
    cap = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, t2, False)
    extrusion.add_boundary(Boundary(cap, solid))
    
    return extrusion
