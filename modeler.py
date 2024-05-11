import numpy as np
from bspy import Solid, Manifold, Boundary

class Modeler:
    def __init__(self):
        self.matrixStack = []
        self.matrix = np.identity(4)
    
    def __call__(self):
        return self.matrix
    
    def pop(self):
        if self.matrixStack:
            self.matrix = self.matrixStack.pop()
        else:
            self.matrix = np.identity(4)

    def push(self):
        self.matrixStack.append(self.matrix)    

    def rotate(self, axis, radians):
        self.matrix = self.matrix @ self.rotation(axis, radians)    
    
    @staticmethod
    def rotation(axis, radians):
        indices = [
            [[1, 1, 2, 2], [1, 2, 1, 2]],
            [[0, 2, 0, 2], [0, 0, 2, 2]],
            [[0, 0, 1, 1], [0, 1, 0, 1]]
        ]
        matrix = np.identity(4)
        matrix[indices[axis][0], indices[axis][1]] = np.array((np.cos(radians), -np.sin(radians), np.sin(radians), np.cos(radians)))
        return matrix

    def scale(self, *args):
        self.matrix = self.matrix @ self.scaling(*args)    
    
    @staticmethod
    def scaling(*args):
        vector = args if np.isscalar(args[0]) else args[0]
        count = min(len(vector), 3)
        matrix = np.identity(4)
        for i in range(count):
            matrix[i, i] = vector[i]
        return matrix

    def transform(self, *args):
        values = args if np.isscalar(args[0]) else args[0]
        if isinstance(values, (Solid, Boundary, Manifold)):
            matrix = self.matrix[:3, :3]
            matrixInverseTranspose = np.transpose(np.linalg.inv(matrix))
            translation = self.matrix[:3, 3]
            if isinstance(values, Solid):
                solid = Solid(values.dimension, values.containsInfinity)
                for boundary in values.boundaries:
                    solid.add_boundary(Boundary(boundary.manifold.transform(matrix, matrixInverseTranspose).translate(translation), boundary.domain))
                return solid
            elif isinstance(values, Boundary):
                return Boundary(values.manifold.transform(matrix, matrixInverseTranspose).translate(translation), values.domain)
            else:
                return values.transform(matrix, matrixInverseTranspose).translate(translation)
        else:
            count = min(len(values), 3)
            vector = np.zeros(4)
            vector[:count] = values[:count]
            vector[3] = 1.0
            return self.matrix @ vector

    def translate(self, *args):
        self.matrix = self.matrix @ self.translation(*args)
        
    @staticmethod
    def translation(*args):
        vector = args if np.isscalar(args[0]) else args[0]
        count = min(len(vector), 3)
        matrix = np.identity(4)
        matrix[:count, 3] = vector[:count]
        return matrix
