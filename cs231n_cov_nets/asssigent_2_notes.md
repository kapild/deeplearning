## Notes from assignemnt #2:


### Reshaping data

Reshape data to:`d_size` is a numebr here 

```
x_reshape = x.reshape(num_inputs, d_size)
```

Or you can reshape to an d1,d2,d3 stype array:

```
d_size = [1,2,3]
x_reshape = x.reshape(num_inputs, *d_size)
``` 

### Loss in backpropagation
loss with respect to any variable U is  nothing but 

```
dL/dU= 
where dL/dy * dy/da * da/du 
= dout * da/du
```

for U being W, the above equation becomes

```
dL/dw= 
where dL/dy * dy/da * da/dw 
= dout * da/dw

da/dw = np.dot(X,W)/dw 
which is X
```

### Loss function

Verify SoftMax and SVM loss equation. [here](https://github.com/kapild/deeplearning/blob/cs231n_cov_nets/cs231n_cov_nets/assignment2/cs231n/layers.py#L506)