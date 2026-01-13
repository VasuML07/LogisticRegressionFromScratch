Logistic Regression From Scratch (NumPy-Based)
Project Overview

This repository contains a from-scratch implementation of Logistic Regression using Python and NumPy only.
No Scikit-learn models. No TensorFlow. No shortcuts.

The purpose of this project is to explain binary classification at the mathematical level, showing how probabilities are computed, how errors are minimized, and how parameters are learned through gradient descent.

This implementation focuses on clarity over abstraction.

Why This Project Matters

Logistic Regression is often treated as a “simple” algorithm.
In reality, it introduces several foundational ideas that appear everywhere in modern machine learning:

Probability modeling

Non-linear activation functions

Loss-based optimization

Gradient descent

Decision boundaries

Understanding this implementation means you understand the backbone of neural networks and classifiers.

What Logistic Regression Does

Logistic Regression predicts binary outcomes (0 or 1) by estimating the probability that an input belongs to a particular class.

Examples:

Spam vs Not Spam

Fraud vs Legitimate

Malignant vs Benign

Pass vs Fail

Instead of predicting raw values, the model predicts probabilities and then converts them into class labels.

High-Level Model Flow
Input Features (X)
        ↓
Linear Combination (z = w·x + b)
        ↓
Sigmoid Activation
        ↓
Probability (0 → 1)
        ↓
Threshold (0.5)
        ↓
Final Class (0 or 1)

Mathematical Foundations
1. Linear Model

The model starts by computing a weighted sum of the input features:

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

Where:

𝑤
w are the learned weights

𝑏
b is the bias

𝑥
x is the input feature vector

This step alone is not enough for classification.

2. Sigmoid Activation Function

To convert the linear output into a probability, the model applies the Sigmoid function:

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


Key properties:

Output range: (0, 1)

Smooth and differentiable

Ideal for probability estimation

Sigmoid Curve (Conceptual)

1.0 ┤            ______
    │          /
0.5 ┤--------•--------
    │       /
0.0 ┼______/__________
           z

3. Binary Cross-Entropy Loss

To measure how wrong the predictions are, the model uses Binary Cross-Entropy Loss:

𝐿
=
−
1
𝑚
∑
𝑖
=
1
𝑚
[
𝑦
𝑖
log
⁡
(
𝑦
^
𝑖
)
+
(
1
−
𝑦
𝑖
)
log
⁡
(
1
−
𝑦
^
𝑖
)
]
L=−
m
1
	​

i=1
∑
m
	​

[y
i
	​

log(
y
^
	​

i
	​

)+(1−y
i
	​

)log(1−
y
^
	​

i
	​

)]

Why this loss?

Strongly penalizes confident wrong predictions

Works naturally with probabilities

Provides smooth gradients for optimization

Lower loss means better predictions.

4. Gradient Descent Optimization

The model learns by updating parameters in the direction that reduces loss.

Gradients

Weight gradient:

∂
𝐿
∂
𝑤
=
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
∂w
∂L
	​

=
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

Bias gradient:

∂
𝐿
∂
𝑏
=
1
𝑚
∑
(
𝑦
^
−
𝑦
)
∂b
∂L
	​

=
m
1
	​

∑(
y
^
	​

−y)
5. Parameter Update Rule
𝑤
:
=
𝑤
−
𝛼
⋅
𝑑
𝑤
w:=w−α⋅dw
𝑏
:
=
𝑏
−
𝛼
⋅
𝑑
𝑏
b:=b−α⋅db

Where:

𝛼
α is the learning rate

This process is repeated over multiple iterations until convergence.

Training Workflow

Initialize weights and bias to zero

Compute linear predictions

Apply sigmoid activation

Calculate loss

Compute gradients

Update parameters

Repeat for fixed iterations

This loop is the engine of learning.

Code Structure Overview

LogisticRegression class

sigmoid(z) – activation function

fit(X, y) – training via gradient descent

predict(X) – class prediction using threshold

The implementation mirrors the math line by line, making it ideal for learning and debugging.

Example Usage
X = [[10,10], [11,15], [12,12], [19,15], [18,20]]
y = [0, 0, 0, 1, 1]

model = LogisticRegression(learning_rate=0.01, iter=500)
model.fit(X, y)

prediction = model.predict([[17.5, 22.0]])

Output Interpretation

Model outputs probabilities internally

A threshold of 0.5 is applied:

Probability ≥ 0.5 → Class 1

Probability < 0.5 → Class 0

This creates a linear decision boundary in feature space.

Learning Outcomes

This project builds strong intuition for:

Probability-based classification

Optimization using gradients

Loss-driven learning

Foundations of neural networks

Why sigmoid + cross-entropy works so well

Everything here scales directly to deep learning.

Requirements

Python 3.x

NumPy

Nothing else.

Future Improvements

Add loss tracking and visualization

Extend to multi-class classification (Softmax)

Implement regularization

Compare with Scikit-learn output
