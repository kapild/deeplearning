### Pandas Notes

- Norm calculation 

	```
	np.linalg.norm(diff_element)
	```
	
- `Sort an array `:
 - Sorts an array by retruning indices.
	```
	np.argsort(dists[i,:])[:k]	
	```

- Most frequent number
	- Does the bin count of each number
	
	```
 	  counts = np.bincount(closest_y)
	  y_pred[i] = np.argmax(counts)

	```
- Get norm of an array on one axis:

	```
	np.square(np.linalg.norm(X, axis=1))
	```

- Duplicate the arrays across matrix
	
	```
	np.array([X_test_norm] * num_train)	
	```

- Get indices of zero elements from the list
	
	```
    idxs = np.flatnonzero(y_train == y)
	```

- Get "n" random indices fom list of indices
	
	```
    idxs = np.random.choice(idxs, samples_per_class, replace=False)	```


