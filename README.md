 # Machine Learning Library 
 
 A machine learning library with commonly used supervised and unsupervised ML algorithms written from scratch using NumPy
 
 ## Models included
 
 - Linear Regression
 - Logistic Regression
 - Naive Bayes
 - K-Nearest Neighbours
 - DBSCAN
 - K-Means

## Example usage

### Linear Regression:

```python
from Models.Supervised import LinearRegression
import numpy as np

test_data = np.array([[1, 1], [2, 2], [3, 4], [4, 4], [5, 5], [5, 6], [6, 5], [7, 7],
                      [7, 6], [8, 8], [9, 7], [10, 11]])

model = LinearRegression(0, 0, 0.01, 100, test_data)  # w0, w1, learning rate, iterations, dataset 
model.fit()
model.getParams()

prediction = model.predict(8)
print("Prediction: ", prediction)
```
Output:
```
Iteration:  100 
Value of w0:  [0.22013201] 
Value of w1: [0.93967292] 
Loss: [0.69535103] 

Prediction:  [7.73751535]
```

### DBSCAN:

```python
from Models.Unsupervised import Dbscan
import numpy as np

test_data = np.array([[0.9, 1], [2, 1], [2.5, 2], [1.2, 2], [5, 3.5], [5, 5.4], [4, 3.9],
                      [5.5, 4], [5.1, 4.6], [4.1, 3.6], [4.6, 3.9],
                      [8, 7], [5.5, 6], [8, 6.5], [7.5, 7.9], [7.5, 6.8]])


dbscan = Dbscan(2, 2, test_data)  # epsilon radius, minimum points, dataset
dbscan.fit()
dbscan.getParams()
```
Output:
```
Points in cluster  1.0 :
(0.9,1.0), (2.0,1.0), (2.5,2.0), (1.2,2.0)
Points in cluster  2.0 :
(5.0,3.5), (5.0,5.4), (4.0,3.9), (5.5,4.0), (5.1,4.6), (4.1,3.6), (4.6,3.9), (5.5,6.0)
Points in cluster  3.0 :
(8.0,7.0), (8.0,6.5), (7.5,7.9), (7.5,6.8)
```
## Imports:
- NumPy
- SciPy (for the sigmoid function)
