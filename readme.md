Logistic Regression From Scratch (NumPy)

This repository contains a from-scratch implementation of Logistic Regression using Python and NumPy only. The goal is to demonstrate how binary classification works at a mathematical and algorithmic level, without relying on machine learning libraries such as Scikit-learn or TensorFlow.

The implementation explicitly shows how probabilities are computed, how loss is calculated, and how parameters are optimized using gradient descent.

What the Model Does

Logistic Regression predicts binary outcomes (0 or 1) by estimating the probability that an input belongs to a particular class.

The model follows this flow:

Input Features (X)
→ Linear Combination (z = w·x + b)
→ Sigmoid Activation
→ Probability (0 to 1)
→ Threshold (0.5)
→ Final Class Prediction

Core Mathematics
Linear Model
𝑧
=
𝑤
1
𝑥
1
+
𝑤
2
𝑥
2
+
⋯
+
𝑤
𝑛
𝑥
𝑛
+
𝑏
z=w
1
	​

x
1
	​

+w
2
	​

x
2
	​

+⋯+w
n
	​

x
n
	​

+b
Sigmoid Activation
𝜎
(
𝑧
)
=
1
1
+
𝑒
−
𝑧
σ(z)=
1+e
−z
1
	​


This function converts any real number into a probability between 0 and 1.

Binary Cross-Entropy Loss
𝐿
=
−
1
𝑚
∑
[
𝑦
log
⁡
(
𝑦
^
)
+
(
1
−
𝑦
)
log
⁡
(
1
−
𝑦
^
)
]
L=−
m
1
	​

∑[ylog(
y
^
	​

)+(1−y)log(1−
y
^
	​

)]

The loss measures how far the predicted probabilities are from the true labels.

Gradient Descent Updates
𝑤
:
=
𝑤
−
𝛼
⋅
1
𝑚
𝑋
𝑇
(
𝑦
^
−
𝑦
)
w:=w−α⋅
m
1
	​

X
T
(
y
^
	​

−y)
𝑏
:
=
𝑏
−
𝛼
⋅
1
𝑚
∑
(
𝑦
^
−
𝑦
)
b:=b−α⋅
m
1
	​

∑(
y
^
	​

−y)

This process is repeated for multiple iterations until the model converges.

Implementation Details

Weights and bias are initialized to zero

Predictions are computed using a linear model + sigmoid

Gradients are calculated manually

Parameters are updated using gradient descent

Final predictions use a threshold of 0.5

Everything is implemented explicitly to mirror the math.

Example
Input:  [17.5, 22.0]
Output: Class 1


Internally, the model predicts a probability and then converts it into a class label.

Requirements

Python 3.x

NumPy

No other dependencies.

Why This Repo Exists

This project is meant to build real intuition for machine learning by stripping away abstractions. If you understand this implementation, you understand the foundation behind neural networks, classifiers, and modern deep learning systems.
