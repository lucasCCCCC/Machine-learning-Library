import numpy as np


class Kmeans:

    def __init__(self, x, K):
        self.K = K
        self.maxIteration = 250
        self.sample, self.attribute = x.shape

    def setCentroid(self, x):
        centroids = np.zeros((self.K, self.attribute))

        for i in range(self.K):
            centroid = x[np.random.choice(range(self.sample))]
            centroids[i] = centroid

        return centroids

    def cluster(self, x, centroids):
        clusters = [[] for i in range(self.K)]

        for i, point in enumerate(x):
            findCentroid = np.argmin(np.sqrt(np.sum((point - centroids)**2, axis=1)))
            clusters[findCentroid].append(i)

        return clusters

    def setNewCentroid(self, x, clusters):
        centroids = np.zeros((self.K, self.attribute))

        for i, cluster in enumerate(clusters):
            newCentroid = np.mean(x[cluster], axis=0)
            centroids[i] = newCentroid

        return centroids

    def predict(self, cluster):
        y = np.zeros(self.sample)

        for i, cluster in enumerate(cluster):
            for j in cluster:
                y[j] = i

        return y

    def printClusters(self, x, y):
        print("Number of clusters specified: ", self.K, "\n")

        clusters = {}

        for i in range(len(x)):
            clusters.setdefault(int(y[i]+1), [])
            coordinate = "(" + str(x[i][0]) + "," + str(x[i][1]) + ")"
            clusters[int(y[i]+1)].append(coordinate)

        for cluster, point in clusters.items():
            print("Points in cluster ", cluster, ":")
            point = str(point).replace("[", "").replace("]", "").replace("'", "")
            print(point, "\n")


    def fitClusters(self, x):
        centroids = self.setCentroid(x)

        for i in range(self.maxIteration):
            clusters = self.cluster(x, centroids)
            prev = centroids

            if not (centroids - prev).any():
                break

        y = self.predict(clusters)


        self.printClusters(x, y)

        return y


x = np.array([[0.9, 1], [2, 1], [2.5, 2], [1.2, 2], [5, 3.5], [5, 5.4], [4, 3.9],
                      [5.5, 4], [5.1, 4.6], [4.1, 3.6], [4.6, 3.9],
                      [8, 7], [5.5, 6], [8, 6.5], [7.5, 7.9], [7.5, 6.8]])

K = 3

kmeans = Kmeans(x, K)
kmeans.fitClusters(x)