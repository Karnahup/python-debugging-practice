import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import minmax_scale

wine_df = pd.read_csv("wine_dataset.csv")

x = wine_df.drop(columns=["target"])
y = wine_df["target"]

scaled = minmax_scale(x)

k_range = range(1, 31)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)

    scores = cross_val_score(
        knn, scaled, y, cv=10, scoring='accuracy'
    )

    cv_scores.append(scores.mean())

plt.plot(k_range, cv_scores)
plt.xlabel('Number of Neighbors')
plt.ylabel('Cross-Validated Accuracy')
plt.title('K-Nearest Neighbors Performance')
plt.grid()
plt.show()

best_score = max(cv_scores)
best_index = cv_scores.index(best_score)
best_k = k_range[best_index]

print("Best K:", best_k)
print("Best Accuracy:", best_score)
