
"""Custom models used in the project."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
 
 
class KMeansClassifier(BaseEstimator, ClassifierMixin):
    """K-Means (2 clusters) used as a classifier.
 
    Clusters are fitted without labels. Afterwards, the cluster with the higher
    share of positive training labels is called the "loan" cluster. A point is
    predicted 1 if it is closer to that cluster's centroid than to the other one.
    """
 
    def __init__(self, random_state=42):
        self.random_state = random_state
 
    def fit(self, X, y):
        y = np.asarray(y)
        self.km_ = KMeans(n_clusters=2, n_init=10, random_state=self.random_state)
        clusters = self.km_.fit_predict(X)
        pos_rate = [y[clusters == c].mean() if (clusters == c).any() else 0 for c in (0, 1)]
        self.pos_cluster_ = int(np.argmax(pos_rate))
        self.neg_cluster_ = 1 - self.pos_cluster_
        self.classes_ = np.array([0, 1])
        return self
 
    def decision_function(self, X):
        """Positive value means closer to the 'loan' centroid (also used for ROC)."""
        d = self.km_.transform(X)
        return d[:, self.neg_cluster_] - d[:, self.pos_cluster_]
 
    def predict(self, X):
        return (self.decision_function(X) > 0).astype(int)
 
