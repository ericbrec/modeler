import numpy as np
from bspy import Solid, Boundary, Hyperplane

def create_hyperplane(normal, offset):
    normalizedNormal = np.atleast_1d(normal)
    assert len(normalizedNormal) < 3
    normalizedNormal = normalizedNormal / np.linalg.norm(normalizedNormal)
    if len(normalizedNormal) > 1:
        return Hyperplane(normalizedNormal, offset * normalizedNormal, np.transpose(np.array([[normal[1], -normal[0]]])))
    else:
        return Hyperplane(normalizedNormal, offset * normalizedNormal, 0.0)

def hyperplane_domain_from_point(hyperplane, point):
    return np.linalg.inv(hyperplane._tangentSpace.T @ hyperplane._tangentSpace) @ hyperplane._tangentSpace.T @ (point - hyperplane._point)

def create_faceted_solid_from_points(points):
    # create_faceted_solid_from_points only works for dimension 2 so far.
    dimension = 2
    assert len(points) > 2
    assert len(points[0]) == dimension

    solid = Solid(dimension, False)
    previousPoint = np.array(points[len(points)-1])
    for point in points:
        point = np.array(point)
        vector = point - previousPoint
        normal = np.array([-vector[1], vector[0]])
        normal = normal / np.linalg.norm(normal)
        hyperplane = create_hyperplane(normal,np.dot(normal,point))
        domain = Solid(dimension-1, False)
        previousPointDomain = hyperplane_domain_from_point(hyperplane, previousPoint)
        pointDomain = hyperplane_domain_from_point(hyperplane, point)
        if previousPointDomain < pointDomain:
            domain.add_boundary(Boundary(create_hyperplane(-1.0, -previousPointDomain), Solid(dimension-2, True)))
            domain.add_boundary(Boundary(create_hyperplane(1.0, pointDomain), Solid(dimension-2, True)))
        else:
            domain.add_boundary(Boundary(create_hyperplane(-1.0, -pointDomain), Solid(dimension-2, True)))
            domain.add_boundary(Boundary(create_hyperplane(1.0, previousPointDomain), Solid(dimension-2, True)))
        solid.add_boundary(Boundary(hyperplane, domain))
        previousPoint = point

    return solid

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
                extrudedDomain.add_boundary(Boundary(create_hyperplane(-1.0, 0.0), Solid(0, True)))
                extrudedDomain.add_boundary(Boundary(create_hyperplane(1.0, extent), Solid(0, True)))
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

def extrude_time(solidFunction, tValues):
    assert(len(tValues) >= 2)

    # Start with solid cap.
    t = tValues[0]
    solid = solidFunction(t)
    extrusion = Solid(solid.dimension + 1, solid.containsInfinity)
    cap = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, -t, True)
    extrusion.add_boundary(Boundary(cap, solid))

    # Interpolate boundaries along time dimension.
    for tNext in tValues[1:]:
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
            domainPoint[solid.dimension-1] = 0
            domainPath.append(domainPoint)
            domainPoint = np.zeros(solid.dimension)
            domainPoint[solid.dimension-1] = tNext - t
            domainPath.append(domainPoint)
            extrudedDomain = extrude_path(boundary.domain, domainPath)
            # Add extruded boundary
            extrudedHyperplane = Hyperplane(extruded_normal, extruded_point, extruded_tangentSpace)
            extrusion.add_boundary(Boundary(extrudedHyperplane, extrudedDomain))
        
        # Compute next sample.
        t = tNext
        solid = solidFunction(t)

    # End with solid cap.
    cap = Hyperplane.create_axis_aligned(extrusion.dimension, solid.dimension, t, False)
    extrusion.add_boundary(Boundary(cap, solid))
    
    return extrusion
