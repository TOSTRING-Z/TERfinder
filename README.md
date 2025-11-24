# TERfinder: Systems Analysis of Functional Transcriptional and Epigenetic Regulators in Leukemia

## Project Description

TERfinder is a computational framework for predicting Enhancer-Promoter Interaction (EPI) status in the K562 leukemia cell line. It integrates DNA sequence one-hot encoding and histone modification features to train an autoencoder for feature extraction, followed by multiple classical machine learning models (GBDT, Logistic Regression, Random Forest, XGBoost, SVM, AdaBoost) to classify EPI pairs as positive (interacting) or negative (non-interacting). The framework outputs key evaluation metrics (F1-Score, AUC-ROC, AUPRC) and visualization plots to assess model performance, enabling systems-level analysis of transcriptional and epigenetic regulators in leukemia.

## Key Features

- One-hot encoding of enhancer/promoter DNA sequences (hg19 genome assembly)
- Integration of histone modification features for EPI pair characterization
- Autoencoder-based feature learning for high-dimensional omics data
- Comparative analysis of 6 machine learning classifiers for EPI prediction
- Automated generation of AUC-ROC/AUPRC curves and F1-Score statistics
- Balanced data splitting (1:1 positive/negative samples) and cross-validation readiness

## Environment Requirements

### 1. Core Dependencies

Install the required packages via pip:

```bash
pip install numpy==1.21.6 pandas==1.3.5 matplotlib==3.5.3 torch==1.12.1 scikit-learn==1.0.2 xgboost==1.6.2
```

### 2. Hardware Requirements

- **CPU**: Intel i7/Ryzen 7 or higher (≥8 cores recommended)
- **GPU**: NVIDIA GPU with CUDA 11.3+ (optional, for accelerated training)
- **Memory**: ≥16GB RAM (32GB recommended for large sequence datasets)

## Data Preparation

### 1. Required Input Files

Organize input files in the following directory structure (match the paths in the code):

```
TERfinder/
├── Enhancer_Extended_f/
│   └── K562_training_Enhancercor12_Extended_h19_DNASequence.fa  # Enhancer DNA sequences (FASTA)
├── Promoter_f/
│   └── K562_training_Promoter_h19_DNASequence.fa               # Promoter DNA sequences (FASTA)
├── K562_training_pairs_label.txt                               # EPI pair labels (1=positive, -1=negative)
├── K562_training_EPI_Enhancer_Extended_histonefeature_Matrix.txt  # Enhancer histone features (matrix)
└── K562_training_EPI_Promoter_histonefeature_Matrix.txt        # Promoter histone features (matrix)
```

### 2. File Format Specifications

| File Type | Format Description |
|-----------|-------------------|
| FASTA Files | Sequences contain only ACGT characters (uppercase/lowercase accepted; no header lines) |
| Label File | Single-column text file with 1 (positive EPI) and -1 (negative EPI) labels |
| Histone Feature Matrix | Tab-separated text file with rows as EPI pairs and columns as histone modification features |

## Running Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/TERfinder.git
cd TERfinder
```

### 2. Prepare Input Data

Follow the directory structure in Data Preparation to place all input files.

### 3. Run the Full Pipeline

```bash
python TERfinder_K562.py
```

### 4. Key Notes

- The pipeline automatically splits data into training (80%) and test (20%) sets
- GPU acceleration is enabled by default (disable by commenting out cuda() calls if no GPU is available)
- Training typically takes 2–3 hours on a GPU (10+ hours on CPU) for 3000 epochs

## Code Implementation

### 1. Header & Seed Initialization

```python
"""
TERfinder: Systems Analysis of Functional Transcriptional and Epigenetic Regulators in Leukemia
Copyright (C) 2022  Xuxiaoqiang.
K562 Cell Line Analysis
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
from sklearn.metrics import auc
from TERfinder import EPI_AutoEncoders  # Import custom autoencoder model
from sklearn import preprocessing
import random
from pandas import DataFrame as df

torch.manual_seed(2022831)  # Fix torch random seed for reproducibility
random.seed(2022831)        # Fix random seed for data splitting
```

**Purpose**: Define project metadata, import core libraries, and set random seeds to ensure reproducibility.

**Key Note**: `EPI_AutoEncoders` is a custom PyTorch module (must be in the same directory).

### 2. Load Enhancer Sequences

```python
f = open('Enhancer_Extended_f/K562_training_Enhancercor12_Extended_h19_DNASequence.fa')
ls_K562_training_Enhancercor12_Extended_h19_DNASequence = []
for line in f:
    if not line.startswith('>'):  # Skip FASTA headers
        ls_K562_training_Enhancercor12_Extended_h19_DNASequence.append(line.replace('\n', ''))  # Remove newline characters
f.close()
print("Loaded each enhancer sequence into list. Total pairs count:")
print(len(ls_K562_training_Enhancercor12_Extended_h19_DNASequence))
```

**Purpose**: Load enhancer DNA sequences from the FASTA file (skip headers and clean newline characters).

**Output**: Print the total number of enhancer sequences (matches the number of EPI pairs).

### 3. One-Hot Encoding Function

```python
def One_hot_Coding(Str_Sequence):
    sequence_length = len(Str_Sequence)
    one_hot_matrix = np.zeros((4, sequence_length))  # 4 rows (A/C/G/T), L columns (sequence length)
    for index in range(sequence_length):
        if Str_Sequence[index] == "a" or Str_Sequence[index] == "A":
            one_hot_matrix[0][index] = 1
        elif Str_Sequence[index] == "c" or Str_Sequence[index] == "C":
            one_hot_matrix[1][index] = 1
        elif Str_Sequence[index] == "g" or Str_Sequence[index] == "G":
            one_hot_matrix[2][index] = 1
        elif Str_Sequence[index] == "t" or Str_Sequence[index] == "T":
            one_hot_matrix[3][index] = 1
        else:
            continue  # Ignore non-ACGT characters (e.g., N)
    return one_hot_matrix
```

**Purpose**: Convert DNA sequences to 4×L one-hot matrices (A=row 0, C=row 1, G=row 2, T=row 3).

**Handling Ambiguity**: Non-ACGT characters are encoded as all zeros.

### 4. Encode Enhancer Sequences

```python
print("Performing one-hot coding for enhancer sequences.")
ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding = []
for seq in ls_K562_training_Enhancercor12_Extended_h19_DNASequence:
    ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding.append(One_hot_Coding(seq))
print("Total encoded enhancer matrices:")
print(len(ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding))
```

**Purpose**: Apply one-hot encoding to all enhancer sequences.

**Output**: Print the number of encoded enhancer matrices (should match the input sequence count).

### 5. Load & Encode Promoter Sequences

```python
f = open('Promoter_f/K562_training_Promoter_h19_DNASequence.fa')
ls_K562_training_Promoter_h19_DNASequence = []
for line in f:
    if not line.startswith('>'):
        ls_K562_training_Promoter_h19_DNASequence.append(line.replace('\n', ''))
f.close()
print("Loaded each promoter sequence into list. Total pairs count:")
print(len(ls_K562_training_Promoter_h19_DNASequence))

print("Performing one-hot coding for promoter sequences. Total encoded matrices:")
ls_K562_training_Promoter_h19_DNASequence_OnehotCoding = []
for seq in ls_K562_training_Promoter_h19_DNASequence:
    ls_K562_training_Promoter_h19_DNASequence_OnehotCoding.append(One_hot_Coding(seq))
print(len(ls_K562_training_Promoter_h19_DNASequence_OnehotCoding))
print("Data loading completed!")
```

**Purpose**: Repeat the loading and encoding process for promoter sequences.

**Validation**: Ensure the number of enhancer and promoter sequences match (one-to-one EPI pairs).

### 6. Load EPI Labels

```python
filename = 'K562_training_pairs_label.txt'
K562_training_pairs_label = np.loadtxt(filename)
K562_training_pairs_label = list(K562_training_pairs_label)
K562_training_pairs_label_tuple = []
for i in range(len(K562_training_pairs_label)):
    K562_training_pairs_label_tuple.append((i, K562_training_pairs_label[i]))  # (index, label) tuples
K562_training_pairs_label_tuple_df = df(K562_training_pairs_label_tuple)
K562_training_pairs_label_tuple_df.to_csv("K562_training_pairs_label_tuple.csv")
```

**Purpose**: Load EPI labels (1/-1) and convert them to index-label tuples for easy data splitting.

**Output**: Save label tuples to a CSV file for traceability.

### 7. Split Positive/Negative Samples

```python
K562_training_positive_pairs_label = K562_training_pairs_label_tuple[0:2891]  # Positive samples (1.0)
K562_training_positive_pairs_label_df = df(K562_training_positive_pairs_label)
K562_training_positive_pairs_label_df.to_csv("K562_training_positive_pairs_label.csv")
K562_training_negative_pairs_label = K562_training_pairs_label_tuple[2891:]   # Negative samples (-1.0)
K562_training_negative_pairs_label_df = df(K562_training_negative_pairs_label)
K562_training_negative_pairs_label_df.to_csv("K562_training_negative_pairs_label.csv")

# Shuffle samples to avoid order bias
random.shuffle(K562_training_positive_pairs_label)
random.shuffle(K562_training_negative_pairs_label)
```

**Purpose**: Separate positive and negative samples and shuffle them to ensure randomness in data splitting.

**Key Note**: The index 2891 is based on the precomputed number of positive samples; adjust if your dataset differs.

### 8. Train/Test Set Splitting

```python
# Training set (80% positive, 80% negative for 1:1 balance)
Model_K562_training_positive_pairs_label = K562_training_positive_pairs_label[0:2312]
Model_K562_training_negative_pairs_label = K562_training_negative_pairs_label[0:2312]
Model_K562_training_pairs = Model_K562_training_positive_pairs_label + Model_K562_training_negative_pairs_label
print("Training dataset length: 4624")
print(len(Model_K562_training_pairs))
Model_K562_training_pairs_df = df(Model_K562_training_pairs)
Model_K562_training_pairs_df.to_csv("Model_K562_training_pairs.csv")

# Test set (remaining samples)
Model_K562_testing_positive_pairs_label = K562_training_positive_pairs_label[2312:]
Model_K562_testing_negative_pairs_label = K562_training_negative_pairs_label[2312:2891]
Model_K562_testing_pairs = Model_K562_testing_positive_pairs_label + Model_K562_testing_negative_pairs_label
Model_K562_testing_pairs_df = df(Model_K562_testing_pairs)
Model_K562_testing_pairs_df.to_csv("Model_K562_testing_pairs.csv")
print("Test dataset length:")
print(len(Model_K562_testing_pairs))
```

**Purpose**: Split positive and negative samples into training (2312 + 2312 = 4624) and test sets (balanced 1:1 for training).

**Output**: Save split indices to CSV files for reproducibility.

### 9. Load Histone Features

```python
print("Loading histone modification features.")
# Enhancer histone features
K562_training_EPI_Enhancer_Extended_histonefeature_Matrix = np.loadtxt("K562_training_EPI_Enhancer_Extended_histonefeature_Matrix.txt")
# Promoter histone features
K562_training_EPI_Promoter_histonefeature_Matrix = np.loadtxt("K562_training_EPI_Promoter_histonefeature_Matrix.txt")
# Concatenate enhancer and promoter features
Enhancer_Promoter_Histone_features = np.concatenate((K562_training_EPI_Enhancer_Extended_histonefeature_Matrix, K562_training_EPI_Promoter_histonefeature_Matrix), axis=1)
print(Enhancer_Promoter_Histone_features.shape)
np.savetxt("Enhancer_Promoter_Histone_features.txt", Enhancer_Promoter_Histone_features)
```

**Purpose**: Load enhancer and promoter histone modification features and concatenate them into a single matrix.

**Output**: Print the shape of the combined feature matrix (rows = EPI pairs, columns = total histone features).

### 10. Data Loader Function

```python
def load_data(ls_Enhancer_h19_DNASequence_OnehotCoding, ls_Promoter_h19_DNASequence_OnehotCoding, EPIhistonefeature_Matrix, ls_K562_EPI_pairs, BATCHSIZE):
    X = []
    Y = []
    Z = []
    for t in ls_K562_EPI_pairs:
        # Concatenate enhancer and promoter one-hot matrices (along the sequence length dimension)
        X.append(np.concatenate((ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]], ls_Promoter_h19_DNASequence_OnehotCoding[t[0]]), axis=1))
        Y.append(np.concatenate((ls_Enhancer_h19_DNASequence_OnehotCoding[t[0]], ls_Promoter_h19_DNASequence_OnehotCoding[t[0]]), axis=1))
        Z.append(EPIhistonefeature_Matrix[t[0]])  # Corresponding histone features
    
    # Convert to PyTorch tensors
    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(np.array(Y))
    Z = torch.FloatTensor(np.array(Z))
    # Create PyTorch dataset
    torch_dataset = Data.TensorDataset(X, Y, Z)
    # Create data loader for batch processing
    data_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True  # Drop incomplete batches
    )
    return data_loader
```

**Purpose**: Create a PyTorch DataLoader to batch-process EPI data (X = input sequence, Y = target sequence, Z = histone features).

**Key Inputs**: One-hot encoded sequences, histone features, EPI pairs, and batch size.

### 11. Initialize Data Loaders

```python
training_dataloader = load_data(ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding, ls_K562_training_Promoter_h19_DNASequence_OnehotCoding, Enhancer_Promoter_Histone_features, Model_K562_training_pairs, 64)
```

**Purpose**: Initialize the training data loader with a batch size of 64.

**Key Note**: Adjust the batch size based on GPU memory (reduce to 32 if out-of-memory errors occur).

### 12. Autoencoder Training Hyperparameters

```python
EPOCH = 3000  # Total training epochs
LR = 0.0001   # Learning rate
K = 0         # Loss weight (K * similarity loss + reconstruction loss)
EPIpred = EPI_AutoEncoders()  # Initialize custom autoencoder
if torch.cuda.is_available():
    EPIpred.cuda()  # Move model to GPU if available
optimizer = torch.optim.Adam(EPIpred.parameters(), lr=LR)  # Adam optimizer
# Loss functions
loss_func1 = nn.BCEWithLogitsLoss()  # Similarity loss (histone features)
loss_func2 = nn.MSELoss(size_average=True, reduction="mean")  # Reconstruction loss (sequence)
```

**Purpose**: Set hyperparameters for autoencoder training (epochs, learning rate, loss functions).

**Loss Design**: Combined loss (similarity between latent features and histone data + sequence reconstruction).

### 13. Autoencoder Training Loop

```python
print("Starting training.")
for epoch in range(EPOCH):
    train_loss = 0
    for step, (x, train_label, z) in enumerate(training_dataloader):
        # Move data to GPU
        b_x = Variable(x).cuda()
        train_label = Variable(train_label).cuda()
        z = Variable(z).cuda()
        # Forward pass
        xlatten, Yout = EPIpred(b_x)
        # Calculate loss
        loss = K * loss_func1(xlatten, z) + loss_func2(Yout, train_label)
        # Backward pass
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Backpropagation
        optimizer.step()       # Update weights
        train_loss += loss.item()
        # Print loss every 100 steps
        if step % 100 == 0:
            print('Epoch: ', epoch, '| Training loss: %.8f' % loss.data.cpu().numpy())
    # Print epoch loss
    print('Epoch: {}, Training Loss: {:.8f}'.format(epoch, train_loss / len(training_dataloader)))
print("Deep learning training completed. Proceeding to machine learning phase.")
```

**Purpose**: Train the autoencoder for 3000 epochs, print loss metrics, and optimize model weights.

**Key Output**: Per-epoch training loss (monitors convergence).

### 14. Extract Latent Features for ML

```python
EPIpred.eval()  # Set model to evaluation mode
# Extract training features
ML_training_dataloader = load_data(ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding, ls_K562_training_Promoter_h19_DNASequence_OnehotCoding, Enhancer_Promoter_Histone_features, Model_K562_training_pairs, 1)
MLtraining_ls_EPIpred_R = []
for step, (x0, test_label0, z) in enumerate(ML_training_dataloader):
    b_x0 = Variable(x0).cuda()
    xlatten0, Yout0 = EPIpred(b_x0)
    # Extract latent features (detach from GPU and convert to numpy)
    MLtraining_ls_EPIpred_R.append(xlatten0.detach().cpu().numpy().tolist())

# Extract test features
testing_dataloader = load_data(ls_K562_training_Enhancercor12_Extended_h19_DNASequence_OnehotCoding, ls_K562_training_Promoter_h19_DNASequence_OnehotCoding, Enhancer_Promoter_Histone_features, Model_K562_testing_pairs, 1)
MLtesting_ls_EPIpred_R = []
for step, (x, test_label, z) in enumerate(testing_dataloader):
    b_x0 = Variable(x).cuda()
    xlatten, Yout = EPIpred(b_x0)
    MLtesting_ls_EPIpred_R.append(xlatten.detach().cpu().numpy().tolist())
```

**Purpose**: Use the trained autoencoder to extract latent features (xlatten) for training and test sets (batch size = 1 for full feature extraction).

**Key Step**: `eval()` mode disables dropout and batch normalization for stable feature extraction.

### 15. Prepare ML Input Data

```python
# Training data
ML_Xtraining_data = df(sum(MLtraining_ls_EPIpred_R, []))
ML_Xtraining_data.to_csv("ML_Xtraining_data.csv")
ls_Ytraining_data = []
for t in Model_K562_training_pairs:
    ls_Ytraining_data.append(t[1])
ML_Ytraining_data = df(ls_Ytraining_data)

# Test data
ML_Xtesting_data = df(sum(MLtesting_ls_EPIpred_R, []))
ML_Xtesting_data.to_csv("ML_Xtesting_data.csv")
ls_Ytesting_data = []
for t in Model_K562_testing_pairs:
    ls_Ytesting_data.append(t[1])
ML_Ytesting_data = df(ls_Ytesting_data)

# Print data shapes for validation
print("Training data:")
print(ML_Xtraining_data.head())
print(ML_Xtraining_data.values.shape)
print(ML_Ytraining_data.head())
print(ML_Ytraining_data.values.shape)
print("Test data:")
print(ML_Xtesting_data.head())
print(ML_Xtesting_data.values.shape)
print(ML_Ytesting_data.head())
print(ML_Ytesting_data.values.shape)
```

**Purpose**: Convert latent features to pandas DataFrames (compatible with machine learning models) and save to CSV files.

**Validation**: Print data shapes to ensure alignment between features and labels.

### 16. Train Machine Learning Models

#### GBDT

```python
print("Training Gradient Boosting Decision Tree (GBDT).")
gbdt = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=50, subsample=1,
                                  min_samples_split=2, min_samples_leaf=1, max_depth=3,
                                  init=None, random_state=None, max_features=None,
                                  verbose=0, max_leaf_nodes=None, warm_start=False)
gbdt.fit(ML_Xtraining_data, ML_Ytraining_data)
gbdt_train_score = gbdt.score(ML_Xtraining_data, ML_Ytraining_data)
print("GBDT training set accuracy: ", gbdt_train_score)
gbdt_test_pred_proba = gbdt.predict_proba(ML_Xtesting_data)
gbdt_test_pred = gbdt.predict(ML_Xtesting_data)
print("GBDT training completed.")
```

#### Logistic Regression

```python
print("Training Logistic Regression (LR).")
LR = LogisticRegression()
LR.fit(ML_Xtraining_data, ML_Ytraining_data)
lr_score = LR.score(ML_Xtraining_data, ML_Ytraining_data)
print("LR training set accuracy: ", lr_score)
lr_test_pred_proba = LR.predict_proba(ML_Xtesting_data)
lr_test_pred = LR.predict(ML_Xtesting_data)
print("LR training completed.")
```

#### Random Forest

```python
print("Training Random Forest (RF).")
rf = RandomForestClassifier(n_estimators=50)
rf.fit(ML_Xtraining_data, ML_Ytraining_data)
rf_score = rf.score(ML_Xtraining_data, ML_Ytraining_data)
print("RF training set accuracy: ", rf_score)
rf_test_pred = rf.predict(ML_Xtesting_data)
rf_test_pred_proba = rf.predict_proba(ML_Xtesting_data)
print("RF training completed.")
```

#### XGBoost

```python
print("Training XGBoost.")
XGB = xgb.XGBClassifier(objective='binary:logistic')
XGB.fit(ML_Xtraining_data, ML_Ytraining_data)
xgb_score = XGB.score(ML_Xtraining_data, ML_Ytraining_data)
print("XGBoost training set accuracy: ", xgb_score)
xgb_test_pred_proba = XGB.predict_proba(ML_Xtesting_data)
xgb_test_pred = XGB.predict(ML_Xtesting_data)
print("XGBoost training completed.")
```

#### SVM

```python
print("Training Support Vector Machine (SVM).")
svm = svm.SVC(C=1, gamma="scale", degree=3, decision_function_shape='ovr', max_iter=-1, kernel="rbf", probability=True)
svm.fit(ML_Xtraining_data, ML_Ytraining_data)
svm_score = svm.score(ML_Xtraining_data, ML_Ytraining_data)
print("SVM training set accuracy: ", svm_score)
svm_test_pred_proba = svm.predict_proba(ML_Xtesting_data)
svm_test_pred = svm.predict(ML_Xtesting_data)
print("SVM training completed.")
```

#### AdaBoost

```python
print("Training AdaBoost.")
AdaB = AdaBoostClassifier(learning_rate=1, n_estimators=600, algorithm="SAMME.R", random_state=42)
AdaB.fit(ML_Xtraining_data, ML_Ytraining_data)
adab_score = AdaB.score(ML_Xtraining_data, ML_Ytraining_data)
print("AdaBoost training set accuracy: ", adab_score)
adab_test_pred = AdaB.predict(ML_Xtesting_data)
adab_test_pred_proba = AdaB.predict_proba(ML_Xtesting_data)
print("AdaBoost training completed.")
```

**Purpose**: Train 6 classical machine learning models on autoencoder latent features.

**Key Output**: Training accuracy and predicted probabilities for test sets.

### 17. Calculate F1-Scores

```python
ls_F1_Methods = []
ls_F1_Score = []
ls_training_acc = []
ls_testing_acc = []

# GBDT F1
print("GBDT F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, gbdt_test_pred)))
ls_F1_Methods.append("GBDT")
ls_F1_Score.append(f1_score(ML_Ytesting_data, gbdt_test_pred))
ls_training_acc.append(gbdt_train_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, gbdt_test_pred))

# LR F1
print("LR F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, lr_test_pred)))
ls_F1_Methods.append("LogisticRegression")
ls_F1_Score.append(f1_score(ML_Ytesting_data, lr_test_pred))
ls_training_acc.append(lr_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, lr_test_pred))

# RF F1
print("RF F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, rf_test_pred)))
ls_F1_Methods.append("RandomForest")
ls_F1_Score.append(f1_score(ML_Ytesting_data, rf_test_pred))
ls_training_acc.append(rf_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, rf_test_pred))

# XGBoost F1
print("XGBoost F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, xgb_test_pred)))
ls_F1_Methods.append("XGBoost")
ls_F1_Score.append(f1_score(ML_Ytesting_data, xgb_test_pred))
ls_training_acc.append(xgb_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, xgb_test_pred))

# SVM F1
print("SVM F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, svm_test_pred)))
ls_F1_Methods.append("SVM")
ls_F1_Score.append(f1_score(ML_Ytesting_data, svm_test_pred))
ls_training_acc.append(svm_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, svm_test_pred))

# AdaBoost F1
print("AdaBoost F1-Score: {:.4f}".format(f1_score(ML_Ytesting_data, adab_test_pred)))
ls_F1_Methods.append("AdaBoost")
ls_F1_Score.append(f1_score(ML_Ytesting_data, adab_test_pred))
ls_training_acc.append(adab_score)
ls_testing_acc.append(accuracy_score(ML_Ytesting_data, adab_test_pred))
```

**Purpose**: Calculate and store F1-scores and accuracy metrics for all models.

**Output**: Comprehensive performance comparison across all classifiers.

### 18. Performance Visualization

#### AUC-ROC Curves

```python
# Calculate ROC curves and AUC values for each model
##GBDT--gbdt_
gbdt_fpr=[]
gbdt_tpr=[]
gbdt_fpr,gbdt_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,gbdt_test_pred_proba[:,1])
np.savetxt("GBDT_fpr.txt",gbdt_fpr)
np.savetxt("GBDT_tpr.txt",gbdt_tpr)
gbdt_roc_auc=metrics.auc(gbdt_fpr,gbdt_tpr)

##LR--lr_
lr_fpr=[]
lr_tpr=[]
lr_fpr,lr_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,lr_test_pred_proba[:,1])
np.savetxt("LR_fpr.txt",lr_fpr)
np.savetxt("LR_tpr.txt",lr_tpr)
lr_roc_auc=metrics.auc(lr_fpr,lr_tpr)

##RF--rf_
rf_fpr=[]
rf_tpr=[]
rf_fpr,rf_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,rf_test_pred_proba[:,1])
np.savetxt("RF_fpr.txt",rf_fpr)
np.savetxt("RF_tpr.txt",rf_tpr)
rf_roc_auc=metrics.auc(rf_fpr,rf_tpr)

##XGBoost--xgb_
xgb_fpr=[]
xgb_tpr=[]
xgb_fpr,xgb_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,xgb_test_pred_proba[:,1])
np.savetxt("XGBoost_fpr.txt",xgb_fpr)
np.savetxt("XGBoost_tpr.txt",xgb_tpr)
xgb_roc_auc=metrics.auc(xgb_fpr,xgb_tpr)

##SVM--svm_
svm_fpr=[]
svm_tpr=[]
svm_fpr,svm_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,svm_test_pred_proba[:,1])
np.savetxt("SVM_fpr.txt",svm_fpr)
np.savetxt("SVM_tpr.txt",svm_tpr)
svm_roc_auc=metrics.auc(svm_fpr,svm_tpr)

##AdaBoost--adab_
adab_fpr=[]
adab_tpr=[]
adab_fpr,adab_tpr,thresholds = metrics.roc_curve(ML_Ytesting_data,adab_test_pred_proba[:,1])
np.savetxt("AdaBoost_fpr.txt",adab_fpr)
np.savetxt("AdaBoost_tpr.txt",adab_tpr)
adab_roc_auc=metrics.auc(adab_fpr,adab_tpr)

#Plot the AUC curves based on the calculated TPR, FPR, and AUC values for each machine learning model
plt.figure(0).clf()
lw=2
##GBDT--gbdt_
plt.plot(gbdt_fpr,gbdt_tpr,color="r",linestyle="--",lw=lw,label='gbdt_ROC curve(area=%0.4f)'%gbdt_roc_auc)
#plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
##LR--lr_
plt.plot(lr_fpr,lr_tpr,color="g",linestyle="--",lw=lw,label='lr_ROC curve(area=%0.4f)'%lr_roc_auc)
##RF--rf_
plt.plot(rf_fpr,rf_tpr,color="b",linestyle="--",lw=lw,label='rf_ROC curve(area=%0.4f)'%rf_roc_auc)
##XGBoost--xgb_
plt.plot(xgb_fpr,xgb_tpr,color="c",linestyle="--",lw=lw,label='xgb_ROC curve(area=%0.4f)'%xgb_roc_auc)
##SVM--svm_
plt.plot(svm_fpr,svm_tpr,color="m",linestyle="--",lw=lw,label='svm_ROC curve(area=%0.4f)'%svm_roc_auc)
#AdaBoost--adab_
plt.plot(adab_fpr,adab_tpr,color="y",linestyle="--",lw=lw,label='adab_ROC curve(area=%0.4f)'%adab_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic ")
plt.legend(loc="lower right")
plt.savefig("Firgue_AUC.svg")
plt.show()
```

#### Precision-Recall Curves

```python
#Plot the P-R curves and calculate the precision, recall, and AUPRC values for each ML model
##GBDT--gbdt_
gbdt_precision=[]
gbdt_recall=[]
gbdt_precision,gbdt_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,gbdt_test_pred_proba[:,1])
np.savetxt("GBDT_precision.txt",gbdt_precision)
np.savetxt("GBDT_recall.txt",gbdt_recall)
gbdt_p_r=metrics.auc(gbdt_recall,gbdt_precision)

##LR--lr_
lr_precision=[]
lr_recall=[]
lr_precision,lr_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,lr_test_pred_proba[:,1])
np.savetxt("LR_precision.txt",lr_precision)
np.savetxt("LR_recall.txt",lr_recall)
lr_p_r=metrics.auc(lr_recall,lr_precision)

##RF--rf_
rf_precision=[]
rf_recall=[]
rf_precision,rf_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,rf_test_pred_proba[:,1])
np.savetxt("RF_precision.txt",rf_precision)
np.savetxt("RF_recall.txt",rf_recall)
rf_p_r=metrics.auc(rf_recall,rf_precision)

##XGBoost--xgb_
xgb_precision=[]
xgb_recall=[]
xgb_precision,xgb_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,xgb_test_pred_proba[:,1])
np.savetxt("XGBoost_precision.txt",xgb_precision)
np.savetxt("XGBoost_recall.txt",xgb_recall)
xgb_p_r=metrics.auc(xgb_recall,xgb_precision)

##SVM--svm
svm_precision=[]
svm_recall=[]
svm_precision,svm_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,svm_test_pred_proba[:,1])
np.savetxt("SVM_precision.txt",svm_precision)
np.savetxt("SVM_recall.txt",svm_recall)
svm_p_r=metrics.auc(svm_recall,svm_precision)

#AdaBoost--adab_
adab_precision=[]
adab_recall=[]
adab_precision,adab_recall,thresholds_pr=precision_recall_curve(ML_Ytesting_data,adab_test_pred_proba[:,1])
np.savetxt("AdaBoost_precision.txt",adab_precision)
np.savetxt("AdaBoost_recall.txt",adab_recall)
adab_p_r=metrics.auc(adab_recall,adab_precision)

plt.figure(0).clf()
##GBDT--gbdt_
plt.plot(gbdt_recall,gbdt_precision,color="r",linestyle="--",lw=lw,label='gbdt_P_R curve(area=%0.4f)'%gbdt_p_r)
##LR--lr_
plt.plot(lr_recall,lr_precision,color="g",linestyle="--",lw=lw,label='lr_P_R curve(area=%0.4f)'%lr_p_r)
##RF--rf_
plt.plot(rf_recall,rf_precision,color="b",linestyle="--",lw=lw,label='rf_P_R curve(area=%0.4f)'%rf_p_r)
##XGBoost--xgb_
plt.plot(xgb_recall,xgb_precision,color="c",linestyle="--",lw=lw,label='xgb_P_R curve(area=%0.4f)'%xgb_p_r)
##SVM--svm_
plt.plot(svm_recall,svm_precision,color="m",linestyle="--",lw=lw,label='svm_P_R curve(area=%0.4f)'%svm_p_r)
#AdaBoost--adab_
plt.plot(adab_recall,adab_precision,color="y",linestyle="--",lw=lw,label='adab_P_R curve(area=%0.4f)'%adab_p_r)
plt.title("Precision/Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend(loc="lower right")
plt.savefig("Firgue_AUPRC.svg")
plt.show()
```

**Purpose**: Generate comprehensive performance visualizations including ROC curves and Precision-Recall curves for all models.

**Output**: Publication-quality SVG figures comparing model performance.
