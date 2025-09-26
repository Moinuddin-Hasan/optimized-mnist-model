# MNIST Digit Classification with a Highly Optimized CNN

This project is a systematic, three-step approach to building a Convolutional Neural Network (CNN) for MNIST digit classification that meets a strict set of performance and efficiency targets.

## Final Model Performance

*   **Test Accuracy:** **99.45%** (consistent in final epochs)
*   **Total Parameters:** **7,416**
*   **Training Epochs:** **15**

---

## Methodology and Results

The core of the assignment was to follow a disciplined, iterative process. Each step involved defining a target, analyzing the results, and using that analysis to inform the next step.

### Step 1: Baseline Model (`models/model_step1.py`)

*   **Target:** Establish a functional baseline CNN with a standard architecture to validate the training pipeline and identify initial weaknesses.
*   **Result:**
    *   Test Accuracy: 98.95%
    *   Parameters: 71,274
*   **Analysis:** The model worked but had far too many parameters (over 71k vs. the <8k target), primarily due to inefficient fully-connected layers. The accuracy was good but not yet at the target level.

<details>
<summary>Click to view Step 1 Training Logs</summary>

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
... (and so on)
```
</details>

### Step 2: Optimized Architecture (`models/model_step2.py`)

*   **Target:** Drastically reduce the parameter count below 8,000 and improve accuracy by introducing modern CNN techniques.
*   **Result:**
    *   Test Accuracy: 99.37%
    *   Parameters: 7,416
*   **Analysis:** Replacing the fully-connected layers with Global Average Pooling was a massive success, bringing the model within the parameter budget. Adding Batch Normalization and Dropout improved accuracy significantly. The only remaining issue was the training speed; it still took 15+ epochs to reach its peak.

<details>
<summary>Click to view Step 2 Training Logs</summary>

```Epoch 1
Test set: Average loss: 0.1207, Accuracy: 9661/10000 (96.61%)
Epoch 2
Test set: Average loss: 0.0544, Accuracy: 9845/10000 (98.45%)
Epoch 3
Test set: Average loss: 0.0509, Accuracy: 9854/10000 (98.54%)
...
Epoch 12
Test set: Average loss: 0.0216, Accuracy: 9934/10000 (99.34%)
Epoch 13
Test set: Average loss: 0.0229, Accuracy: 9931/10000 (99.31%)
Epoch 14
Test set: Average loss: 0.0226, Accuracy: 9929/10000 (99.29%)
Epoch 15
Test set: Average loss: 0.0205, Accuracy: 9937/10000 (99.37%)
... (and so on)
```
</details>

### Step 3: Advanced Training Techniques (`models/model_step3.py`)

*   **Target:** Achieve the final target of consistent >99.4% accuracy within 15 epochs by using advanced training methods.
*   **Result:**
    *   Test Accuracy: **99.45%** (stable in final epochs)
    *   Parameters: **7,416**
    *   Epochs: **15**
*   **Analysis:** Success. The combination of Data Augmentation and a OneCycleLR scheduler allowed the model to train faster and generalize better, leading to a stable and high accuracy that met all project requirements.

<details>
<summary>Click to view Step 3 Final Logs</summary>

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

## How to Run

1.  Clone the repository.
2.  Set up a Python environment and install dependencies: `pip install -r requirements.txt`
3.  Run any of the experiments:
    ```bash
    # Run the final, optimized model
    python train.py --model model_step3

    # Run the baseline model
    python train.py --model model_step1
    ```