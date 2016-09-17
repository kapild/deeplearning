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

- Making a row vector from numpy

	```
	np.ones((X_train.shape[0], 1))
	```
- Stack a row vector 

	```
	np.hstack([X_train, np.ones((X_train.shape[0], 1))])	
	```
- Radom sample from normal distribution
	
	```
	sigma * np.random.randn(...) + mu		
	
	```
	-  Two-by-four array of samples from N(3, 6.25):
	
		```
		 2.5 * np.random.randn(2, 4) + 3	
		```
	
- Random range

	```
	random.randrange(stop)
	```
- Finding maximum value index in an array
	
	```
	y_pred = np.argmax(X_W_dot, axis=1)
	```
	
	
    	