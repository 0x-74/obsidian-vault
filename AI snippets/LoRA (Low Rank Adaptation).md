- instead of training the weights the changes are tracked in two separate lower ranked (dimensionality) matrix 
- these lower ranked matrix when multiplied result in the original dimension of the weights of the LLM.
- these can then be added to the original weights directly
- increasing rank increases precision, rank denotes linear independence not dimensionality
below is a rank 1 matrix: 
$$ \begin{array}{cc} 1 &2 \\ 1 &2 \end{array} $$ 
below is rank 2 matrix:
$$ \begin{array}{cc} 1&2 \\ 3&4 \end{array}$$
- this is way more efficient as the number of parameters trained are drastically reduced even for relatively high ranks
- low ranks work just as well as high ranks for non complex tasks
- complex tasks are task outside the scope of which a model is trained for example giving dieting tips when a model was trained to only give gym instructions and not health advice