# Debugging: KNN Cross-Validation Program

## Problem
I was implementing K-Nearest Neighbors (KNN)
using cross-validation on the Wine dataset.

I encountered two issues in my program:

1. I used the original feature data instead of
   the scaled data during cross-validation.
2. I placed the plotting code inside the for loop,
   causing the graph to be displayed repeatedly.

## Cause
KNN is distance-based, so feature scaling is important.
Also, the graph should be plotted after all K values
have been evaluated.

## Solution
1. Used the scaled dataset in cross_val_score().
2. Moved the plotting statements outside the for loop.
3. Calculated the best K value using the highest
   cross-validation accuracy.

## What I learned
- Why feature scaling matters in KNN.
- How cross-validation evaluates model performance.
- How indentation affects Python program execution.
- How to debug and improve machine learning code.

## Tools used
- Python
- Pandas
- Scikit-learn
- Matplotlib
