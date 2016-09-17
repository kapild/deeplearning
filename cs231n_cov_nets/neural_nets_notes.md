### Neural nets notes

- The 2 layer net is made of the following arch
	- Input layer with D dimension
	- First hidden layer with H dimensions
	- Output layer with C dimensions

- Weight matrices
	- 	Given the above structures weights can be defined as
	-  W1 = D x H
	-  W2 = H x C
	-  Note: here C are the total number of output class. 
	-  b1 : first bias layer of size 1 x H
	-  b2: second bias layer with size 1 x C

- Steps
	- Dot product of input X with W1 (including bias)
	- ReLu actiivation
	- dot product of activation with w2 (inclduing weights)

