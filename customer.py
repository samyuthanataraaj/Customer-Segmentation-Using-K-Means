import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from mpl_toolkits.mplot3d import Axes3D


df = pd.read_csv(r"D:\tamilanskills\project3\customer.csv")   

print("First 5 rows of dataset:")
print(df.head())


categorical_cols = ["gender", "education", "country"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


features = ["age", "income", "purchase_frequency", "spending"]
X = df[features]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


inertia = []
K_range = range(1, 10)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.show()


kmeans = KMeans(n_clusters=4, random_state=42, n_init=10) 
df["Cluster"] = kmeans.fit_predict(X_scaled)

print("\nCluster Centers (scaled):")
print(kmeans.cluster_centers_)

print("\nClustered Data Sample:")
print(df.head())


plt.figure(figsize=(8,6))
sns.scatterplot(x=df["income"], y=df["spending"], hue=df["Cluster"], palette="viridis")
plt.title("Customer Segmentation (Income vs Spending)")
plt.show()

fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df["age"], df["income"], df["spending"], c=df["Cluster"], cmap="viridis", s=50)
ax.set_xlabel("Age")
ax.set_ylabel("Income")
ax.set_zlabel("Spending")
ax.set_title("3D Customer Segmentation")
plt.show()
