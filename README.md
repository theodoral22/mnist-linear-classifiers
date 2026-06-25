# Linear Classifiers for MNIST: Perceptron (OvO, OvA) & Softmax

An implementation of linear classification algorithms from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## 📌 Project Overview
The goal of this project is to implement, evaluate, and compare three multi-class linear classification models without relying on high-level machine learning libraries (like `sklearn.linear_model` or `PyTorch`). 

The implemented algorithms are:
1. **Perceptron One-versus-One (OvO)** (45 binary classifiers)
2. **Perceptron One-versus-All (OvA)** (10 binary classifiers)
3. **Linear Classifier with Softmax** (Cross-Entropy Loss optimized via Stochastic Gradient Descent)

## 📊 Experimental Setup
The models were trained and evaluated under two distinct scenarios to observe how class imbalances affect linear decision boundaries:
* **Scenario A (Balanced):** 1,000 training samples per class.
* **Scenario B (Imbalanced):** 1,000 training samples for the majority class (Digit '5') and only 50 samples for each of the remaining 9 classes.

## 🧠 Key Findings
Detailed mathematical analysis and evaluation metrics can be found in the [Project Report](mnist_linear_classifiers_report.pdf). Key takeaways include:

* **Perceptron Learning Rate Invariance:** Under zero-weight initialization, the learning rate (alpha) acts merely as a scalar multiplier. It changes the magnitude but not the direction of the weight vector, resulting in identical decision boundaries and identical learning curves regardless of the learning rate used (e.g., a=10 vs a=0.0001).
* **OvO vs. OvA:** The One-versus-One strategy geometrically simplifies the classification space, consistently outperforming the One-versus-All approach in linearly non-separable datasets. OvO achieved a max accuracy of **91.03%**, while OvA reached **86.83%**.
* **Impact of Imbalance (Softmax):** In Scenario B, the Softmax classifier developed a strong probabilistic bias toward the majority class ('5'). This was clearly reflected in the confusion matrices, where the model systematically predicted '5' in moments of uncertainty.
