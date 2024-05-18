import numpy as np
from bspy import Solid, Manifold, Boundary

class Modeler:
    def __init__(self):
        self.matrixStack = []
        self.matrix = np.identity(4)
        self.dMatrix = np.zeros((4, 4))
        self.dMatrix[3,3] = 1.0
    
    def __call__(self):
        return self.matrix
    
    def pop(self):
        if self.matrixStack:
            self.matrix, self.dMatrix = self.matrixStack.pop()
        else:
            self.matrix = np.identity(4)
            self.dMatrix = np.zeros((4, 4))
            self.dMatrix[3,3] = 1.0

    def push(self):
        self.matrixStack.append((self.matrix, self.dMatrix))

    def multiply(self, matrix, dMatrix):
        self.matrix = self.matrix @ matrix
        self.dMatrix[:3,:3] = self.dMatrix[:3,:3] @ matrix[:3,:3] + self.matrix[:3,:3] @ dMatrix[:3,:3]
        self.dMatrix[:3,3] += self.dMatrix[:3,:3] @ matrix[:3, 3] + self.matrix[:3,:3] @ dMatrix[:3, 3]

    def rotate(self, axis, radians, dRadians=0.0):
        self.multiply(*self.rotation(axis, radians, dRadians))
    
    @staticmethod
    def rotation(axis, radians, dRadians=0.0):
        indices = [
            [[1, 1, 2, 2], [1, 2, 1, 2]],
            [[0, 2, 0, 2], [0, 0, 2, 2]],
            [[0, 0, 1, 1], [0, 1, 0, 1]]
        ]
        matrix = np.identity(4)
        matrix[indices[axis][0], indices[axis][1]] = np.array((np.cos(radians), -np.sin(radians), np.sin(radians), np.cos(radians)))

        dMatrix = np.zeros((4, 4))
        dMatrix[3,3] = 1.0
        dMatrix[indices[axis][0], indices[axis][1]] = dRadians * np.array((-np.sin(radians), -np.cos(radians), np.cos(radians), -np.sin(radians)))

        return matrix, dMatrix

    def scale(self, v, dV=(0.0, 0.0, 0.0)):
        self.multiply(*self.scaling(v, dV))
    
    @staticmethod
    def scaling(v, dV=(0.0, 0.0, 0.0)):
        matrix = np.identity(4)
        for i in range(min(len(v), 3)):
            matrix[i, i] = v[i]

        dMatrix = np.zeros((4, 4))
        dMatrix[3,3] = 1.0
        for i in range(min(len(dV), 3)):
            dMatrix[i, i] = dV[i]

        return matrix, dMatrix

    def transform(self, *args):
        values = args if np.isscalar(args[0]) else args[0]
        if isinstance(values, (Solid, Boundary, Manifold)):
            matrix = self.matrix[:3, :3]
            matrixInverseTranspose = np.transpose(np.linalg.inv(matrix))
            translation = self.matrix[:3, 3]
            def transformManifold(manifold):
                transformedManifold = manifold.transform(matrix, matrixInverseTranspose).translate(translation)
                transformedManifold._dP = self.dMatrix[:3, :3] @ manifold.evaluate((0.0, 0.0)) + self.dMatrix[:3, 3]
                return transformedManifold
            
            if isinstance(values, Solid):
                solid = Solid(values.dimension, values.containsInfinity)
                for boundary in values.boundaries:
                    solid.add_boundary(Boundary(transformManifold(boundary.manifold), boundary.domain))
                return solid
            elif isinstance(values, Boundary):
                return Boundary(transformManifold(values.manifold), values.domain)
            else:
                return transformManifold(values)
        else:
            count = min(len(values), 3)
            vector = np.zeros(4)
            vector[:count] = values[:count]
            vector[3] = 1.0
            return self.matrix @ vector

    def translate(self, v, dV=(0.0, 0.0, 0.0)):
        self.multiply(*self.translation(v, dV))
        
    @staticmethod
    def translation(v, dV=(0.0, 0.0, 0.0)):
        count = min(len(v), 3)
        matrix = np.identity(4)
        matrix[:count, 3] = v[:count]

        count = min(len(dV), 3)
        dMatrix = np.zeros((4, 4))
        dMatrix[3,3] = 1.0
        dMatrix[:count, 3] = dV[:count]

        return matrix, dMatrix
