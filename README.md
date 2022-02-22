# k-means
simple implementation of k-means clustering 

This is a very simple Python package which performs k-means clustering. 
In particular, this package allows one to manually set "fixed" cluster centers and optimize the rest of the clusters around them,
a feature I couldn't find in any existing k-means clustering code. 

Example usage:
```
crd = [...]
predefined_centers = [...]

km = kmeans.KMeans(n_clusters=20)
km.fit(crds, predefined_centers)

print(km.centers)
```

Corin Wagen, 2022
