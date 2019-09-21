
# coding: utf-8

# In[ ]:


from sklearn.externals import joblib
scaler = joblib.load('scaler.pkl')
iris_clustering = joblib.load('iris_clustering.pkl')
from scipy.spatial.distance import euclidean
def auto_clustering(sepal_length,sepal_width, petal_length, petal_width):
    x = [sepal_length,sepal_width, petal_length, petal_width]
    x_mean = scaler.mean_
    x_var = scaler.var_
    for i in range(4):
        x[i] = (x[i]-x_mean[i])/x_var[i]
    #if you don't wanna customize the distance, the following command will be enough
    #return(iris_clustering.predict([x]))
    
    center_0 = iris_clustering.cluster_centers_[0]
    center_1 = iris_clustering.cluster_centers_[1]    
    center_2 = iris_clustering.cluster_centers_[2]
    dist = {
    'Cluster 0' : euclidean(center_0, x),
    'Cluster 1' : euclidean(center_1, x),
    'Cluster 2' : euclidean(center_2, x)
    }
    return(min(dist, key=dist.get))

if __name__ == '__main__':
    sepal_length = float(input('Sepal length : '))
    sepal_width = float(input('Sepal width : '))
    petal_length = float(input('Petal length : '))
    petal_width = float(input('Petal width : '))
    print(auto_clustering(sepal_length,sepal_width, petal_length, petal_width))

