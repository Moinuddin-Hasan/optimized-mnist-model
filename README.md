# MNIST Digit Classification with a Highly Optimized CNN

This project documents a systematic, three-step approach to building a Convolutional Neural Network (CNN). The goal was not simply to classify MNIST digits, but to do so under a strict set of constraints: high accuracy, a low parameter count, and rapid training. This repository serves as a case study in methodical machine learning engineering, moving from a naive baseline to a highly optimized and efficient final model.

## Final Model Performance

*   **Test Accuracy:** **99.45%** (consistent in final epochs)
*   **Total Parameters:** **7,416**
*   **Training Epochs:** **15**

---

## Project Structure

The repository is organized into a modular and clean structure, separating model definitions from the training logic.

```
mnist_assignment/
├── models/
│   ├── model_step1.py   # Inefficient baseline model
│   ├── model_step2.py   # Architecturally optimized model
│   └── model_step3.py   # Final model (same architecture, advanced training)
├── train.py             # Main script for training and evaluation
├── requirements.txt     # Project dependencies
└── README.md            # Project documentation and analysis
```

---

## Methodology and Results

The core of the assignment was to follow a disciplined, iterative process. Each step involved defining a target, analyzing the results, and using that analysis to justify the next set of improvements.

### Step 1: Baseline Model (`models/model_step1.py`)

*   **Target:** Establish a functional baseline CNN with a standard, well-understood architecture. The goal was to validate the training pipeline and produce an initial result to analyze for weaknesses.
*   **Result:**
    *   Test Accuracy: 98.95%
    *   Parameters: 71,274
*   **Analysis:** The model worked and achieved a respectable accuracy, proving the pipeline was functional. However, it failed the core constraints significantly. The 71k parameters were nearly nine times the target limit. The culprit was the inefficient fully-connected layers, which create a dense mapping of every feature map pixel to the output neurons, leading to a parameter explosion.

<details>
<summary><strong>View Full Step 1 Training Logs</strong></summary>

```
Epoch 1
Test set: Average loss: 0.0939, Accuracy: 9717/10000 (97.17%)
Epoch 2
Test set: Average loss: 0.0655, Accuracy: 9794/10000 (97.94%)
Epoch 3
Test set: Average loss: 0.0510, Accuracy: 9835/10000 (98.35%)
Epoch 4
Test set: Average loss: 0.0508, Accuracy: 9840/10000 (98.40%)
Epoch 5
Test set: Average loss: 0.0504, Accuracy: 9838/10000 (98.38%)
Epoch 6
Test set: Average loss: 0.0399, Accuracy: 9876/10000 (98.76%)
Epoch 7
Test set: Average loss: 0.0459, Accuracy: 9860/10000 (98.60%)
Epoch 8
Test set: Average loss: 0.0387, Accuracy: 9891/10000 (98.91%)
Epoch 9
Test set: Average loss: 0.0346, Accuracy: 9895/10000 (98.95%)
Epoch 10
Test set: Average loss: 0.0446, Accuracy: 9865/10000 (98.65%)
Epoch 11
Test set: Average loss: 0.0399, Accuracy: 9895/10000 (98.95%)
Epoch 12
Test set: Average loss: 0.0465, Accuracy: 9882/10000 (98.82%)
Epoch 13
Test set: Average loss: 0.0502, Accuracy: 9875/10000 (98.75%)
Epoch 14
Test set: Average loss: 0.0526, Accuracy: 9871/10000 (98.71%)
Epoch 15
Test set: Average loss: 0.0469, Accuracy: 9888/10000 (98.88%)
Epoch 16
Test set: Average loss: 0.0521, Accuracy: 9868/10000 (98.68%)
Epoch 17
Test set: Average loss: 0.0548, Accuracy: 9871/10000 (98.71%)
Epoch 18
Test set: Average loss: 0.0528, Accuracy: 9872/10000 (98.72%)
Epoch 19
Test set: Average loss: 0.0503, Accuracy: 9893/10000 (98.93%)
Epoch 20
Test set: Average loss: 0.0551, Accuracy: 9890/10000 (98.90%)
```
</details>

### Step 2: Optimized Architecture (`models/model_step2.py`)

*   **Target:** Drastically reduce the parameter count below the 8,000 limit and improve accuracy by introducing modern, efficient CNN techniques.
*   **Result:**
    *   Test Accuracy: 99.37%
    *   Parameters: 7,416
*   **Analysis:** This step marked a major architectural overhaul. Replacing the fully-connected layers with **Global Average Pooling (GAP)** was the key to solving the parameter problem, reducing the count by an order of magnitude. Concurrently, **Batch Normalization** was added to stabilize gradients and accelerate training, while **Dropout** provided essential regularization to combat the overfitting seen in Step 1. The model was now efficient and highly accurate, but its training speed was still a concern.

<details>
<summary><strong>View Full Step 2 Training Logs</strong></summary>

```
Epoch 1
Test set: Average loss: 0.1207, Accuracy: 9661/10000 (96.61%)
Epoch 2
Test set: Average loss: 0.0544, Accuracy: 9845/10000 (98.45%)
Epoch 3
Test set: Average loss: 0.0509, Accuracy: 9854/10000 (98.54%)
Epoch 4
Test set: Average loss: 0.0371, Accuracy: 9891/10000 (98.91%)
Epoch 5
Test set: Average loss: 0.0357, Accuracy: 9896/10000 (98.96%)
Epoch 6
Test set: Average loss: 0.0321, Accuracy: 9902/10000 (99.02%)
Epoch 7
Test set: Average loss: 0.0276, Accuracy: 9917/10000 (99.17%)
Epoch 8
Test set: Average loss: 0.0271, Accuracy: 9923/10000 (99.23%)
Epoch 9
Test set: Average loss: 0.0288, Accuracy: 9905/10000 (99.05%)
Epoch 10
Test set: Average loss: 0.0229, Accuracy: 9919/10000 (99.19%)
Epoch 11
Test set: Average loss: 0.0251, Accuracy: 9919/10000 (99.19%)
Epoch 12
Test set: Average loss: 0.0216, Accuracy: 9934/10000 (99.34%)
Epoch 13
Test set: Average loss: 0.0229, Accuracy: 9931/10000 (99.31%)
Epoch 14
Test set: Average loss: 0.0226, Accuracy: 9929/10000 (99.29%)
Epoch 15
Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)
Epoch 16
Test set: Average loss: 0.0236, Accuracy: 9927/10000 (99.27%)
Epoch 17
Test set: Average loss: 0.0206, Accuracy: 9932/10000 (99.32%)
Epoch 18
Test set: Average loss: 0.0208, Accuracy: 9932/10000 (99.32%)
Epoch 19
Test set: Average loss: 0.0204, Accuracy: 9937/10000 (99.37%)
Epoch 20
Test set: Average loss: 0.0220, Accuracy: 9929/10000 (99.29%)
```
</details>

### Step 3: Advanced Training Techniques (`models/model_step3.py`)

*   **Target:** Achieve the final target of consistent >99.4% accuracy within 15 epochs by shifting focus from architecture to advanced training methods.
*   **Result:**
    *   Test Accuracy: **99.45%** (stable in final epochs)
    *   Parameters: **7,416**
    *   Epochs: **15**
*   **Analysis:** Success. With a solid architecture in place, this phase focused on optimizing the training dynamics. The **OneCycleLR scheduler** intelligently managed the learning rate to achieve faster convergence, while **Data Augmentation** made the model more robust. This combination was critical for pushing the accuracy past the 99.4% threshold and keeping it there consistently, all within the 15-epoch time limit.

<details>
<summary><strong>View Full Step 3 Final Logs</strong></summary>

```
Epoch 1
Test set: Average loss: 0.1744, Accuracy: 9515/10000 (95.15%)
Epoch 2
Test set: Average loss: 0.0589, Accuracy: 9838/10000 (98.38%)
Epoch 3
Test set: Average loss: 0.0372, Accuracy: 9895/10000 (98.95%)
Epoch 4
Test set: Average loss: 0.0436, Accuracy: 9864/10000 (98.64%)
Epoch 5
Test set: Average loss: 0.0265, Accuracy: 9918/10000 (99.18%)
Epoch 6
Test set: Average loss: 0.0263, Accuracy: 9912/10000 (99.12%)
Epoch 7
Test set: Average loss: 0.0318, Accuracy: 9898/10000 (98.98%)
Epoch 8
Test set: Average loss: 0.0254, Accuracy: 9917/10000 (99.17%)
Epoch 9
Test set: Average loss: 0.0221, Accuracy: 9922/10000 (99.22%)
Epoch 10
Test set: Average loss: 0.0230, Accuracy: 9925/10000 (99.25%)
Epoch 11
Test set: Average loss: 0.0190, Accuracy: 9942/10000 (99.42%)
Epoch 12
Test set: Average loss: 0.0184, Accuracy: 9934/10000 (99.34%)
Epoch 13
Test set: Average loss: 0.0171, Accuracy: 9945/10000 (99.45%)
Epoch 14
Test set: Average loss: 0.0171, Accuracy: 9943/10000 (99.43%)
Epoch 15
Test set: Average loss: 0.0152, Accuracy: 9944/10000 (99.44%)
```
</details>

---

## Conclusion

This project successfully demonstrates that achieving high performance in deep learning is not just about complex architectures, but about a disciplined and analytical approach. By systematically identifying bottlenecks and applying the correct techniques in the correct order—from architectural optimization to advanced training strategies—all performance targets were successfully met.

---

## How to Run This Project

### Prerequisites
*   Python 3.11
*   PyTorch and related libraries (as listed in the requirements.txt

### Setup and Execution
1.  Clone the repository to your local machine.
2.  Set up a Python environment (conda or venv is recommended) and install all dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run all of the three models from the command line:
    ```bash
    # Run the initial baseline model (Step 1)
    python train.py --model model_step1
    
    # Run the optimized architecture model (Step 2)
    python train.py --model model_step2
    
    # Run the final, optimized model (Step 3)
    python train.py --model model_step3
    ```
