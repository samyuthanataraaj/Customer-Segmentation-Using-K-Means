

## Problem Statement

Businesses need to target different customers differently based on their behavior. A one-size-fits-all approach in marketing is inefficient and costly.

## Objective

The goal of this project is to **cluster customers into groups** using **unsupervised learning (K-Means clustering)**, based on their age, income, purchase frequency, and spending patterns.

## Requirements

* Python 3.x
* Libraries:

  * `pandas` → Data handling
  * `numpy` → Numerical operations
  * `matplotlib` & `seaborn` → Visualization
  * `scikit-learn` → Preprocessing & K-Means clustering

## Dataset

The dataset contains the following columns:

* `name` → Customer name
* `age` → Age of the customer
* `gender` → Male/Female/Other
* `education` → Education level
* `income` → Annual income
* `country` → Customer’s country
* `purchase_frequency` → Number of purchases in a given time
* `spending` → Amount spent by the customer

## Steps Followed

1. **Load Dataset** → Import the CSV file.
2. **Preprocessing**

   * Handle missing values.
   * Encode categorical columns if needed (`gender`, `education`, `country`).
   * Scale numerical features (`age`, `income`, `purchase_frequency`, `spending`).
3. **Apply K-Means Clustering**

   * Use the **Elbow Method** to find the optimal number of clusters.
   * Fit K-Means with selected cluster count (3–5 clusters).
4. **Visualization**

   * Plot 2D scatter plots (Age vs Income, Income vs Spending, etc.) with cluster coloring.
   * (Optional) Use PCA for 3D visualization.
5. **Interpret Results**

   * Identify meaningful customer segments like *High Income–High Spending*, *Low Income–Frequent Buyers*, etc.

## Expected Outcome

* 3–5 meaningful customer groups for targeted marketing.
* Visual representation of clusters to understand patterns.

## Usage

```bash
# Clone repo
git clone <your-repo-link>

# Install dependencies
pip install -r requirements.txt

# Run project
python customer_segmentation.py
```

## Example Insights

* Cluster 1: Young, low income, frequent buyers
* Cluster 2: Middle-aged, high income, moderate spending
* Cluster 3: Older, high income, high spending


