### Supervised ML
known inputs and outputs are used for train input X -> output Y

represent txt as vec
create array of vocab consists of all unique/new words
a representation in 0's and 1's where most values are 0 is called a sparse representation 
training logistic regression on sparse representation is a bad idea because n+1 vocab size parameters are needed so computation becomes expensive with vocabulary
make frequency table where frequency of each word in each class label is stored (eg the amt of time the word occurs in posive laitbel total and the total times negative Freq occurs) we do thisd to make training easy as it will train on only 3 features now , output label and negative and positive frequency counts


 $X_m = [label(1 \ or \ 0),\sum_w freqs(w,1),\sum_w freqs(w,0)]$
 where w,1 represents summation of positive labelled vocabulary that appears in the input
 and w,0 represents negative or Y=0 frequencies that occur in the input sentence
for preprocessing we remove stop words (is,am,are) and punctuations but punctuations might be necessary depending on the NLP task. we also perform stemming converting a word into its stem example tun is the stem for tuning tuned and tune and take care of Upper and Lower case letters merging them into 1. now we keep stacking this 3 dimensional array where each row represents one training example 

Logistic regression 
$P(Y = 1 \mid \mathbf{X}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T \mathbf{X}}}$
Accuracy Formula
$\text{Accuracy} = \frac{1}{m} \sum_{i=1}^{m} \mathbf{1}\left( \hat{y}^{(i)} = y^{(i)} \right)$
Where $1$ is the indicator function, which returns $1$ if the argument is true, and $0$ otherwise.

![[Pasted image 20250513003242.png]]

either $y^{(i)}$ or ${1-y^{(i)}}$  will give loss because prediction is always 1 or 0 so one of the terms multiplies by 0 and cancels out. we take -ve of 1/m because log between 1 to 0 is always negative. if the prediction and actual label dont match the losses blow up hence giving us high loss for wrong predictions
this cost function is also called Maximum Likelihood Estimation (MLE)
