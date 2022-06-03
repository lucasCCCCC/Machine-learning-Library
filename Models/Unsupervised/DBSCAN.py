import numpy as np
import queue


class Dbscan:

    def __init__(self, epsilon, minPoints, testData):
        self.epsilon = epsilon
        self.minPoints = minPoints
        self.noise = 0
        self.clusters = 0
        self.test_data = np.array(testData)

    def euclideanDistance(self, p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def findNeighbourhoodPoints(self, x):
        neighbours = []

        for i in range(len(self.test_data)):
            if self.euclideanDistance(x, self.test_data[i, :2]) <= self.epsilon:
                neighbours.append(i)

        return neighbours

    def fit(self):
        neighboursQueue = queue.Queue()

        self.test_data = np.append(self.test_data, np.array([[-1] * len(self.test_data)]).reshape(-1, 1), axis=1)

        for i in range(len(self.test_data)):

            if self.test_data[i, 2] != -1:
                continue

            findNeighbours = self.findNeighbourhoodPoints(self.test_data[i, :2])

            if len(findNeighbours) <= self.minPoints:
                self.test_data[i, 2] = self.noise
                continue

            self.clusters += 1
            self.test_data[i, 2] = self.clusters

            foundNeighbours = findNeighbours

            for i in findNeighbours:
                neighboursQueue.put(i)

            while not neighboursQueue.empty():
                point = neighboursQueue.get()

                if self.test_data[point, 2] == 0:
                    self.test_data[point, 2] = self.clusters

                if self.test_data[point, 2] != -1:
                    continue

                self.test_data[point, 2] = self.clusters

                point2 = self.test_data[point, :2]
                neighbourPoints = self.findNeighbourhoodPoints(point2)

                if len(neighbourPoints) >= self.minPoints:
                    for i in neighbourPoints:
                        if i not in foundNeighbours:
                            neighboursQueue.put(i)
                            foundNeighbours.append(i)

    def getParams(self):

        clusters = {}

        for x in self.test_data:
            clusters.setdefault(str(x[2]), [])
            coordinate = "(" + str(x[0]) + "," + str(x[1]) + ")"
            clusters[str(x[2])].append(coordinate)

        for cluster, point in clusters.items():
            print("Points in cluster ", cluster, ":")
            point = str(point).replace("[", "").replace("]", "").replace("'", "")
            print(point)
