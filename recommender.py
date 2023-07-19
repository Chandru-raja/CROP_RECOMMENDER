
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

dataset = pd.read_csv("Crop_recommendation.csv")
X = dataset.iloc[:,:-2].values
y = dataset.iloc[:,-1]
le = LabelEncoder()
print(y)
y= le.fit_transform(dataset['label'])
print(y)
print("x:\n",X)
print("\ny:\n",max(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=15)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("\ny_pred:\n",y_pred)
print("\ny_test:\n",y_test)

cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)
import seaborn as sns
sns.heatmap(cm, annot=True)
plt.show()
print("\nCONFUSION MATRIX:\n",cm)
print("\nACCURACY:",ac*100,"%")


print("ENTER THE INPUTS:(90,38,39,20,87,200)")
n =int(input("N: "))
p =int(input("P: "))
k =int(input("K: "))
temp =int(input("TEMPERATURE: "))
hum =int(input("HUMIDITY: "))
ph =int(input("PH: "))
print("RECOMMENDED CROP: ",str(le.inverse_transform(knn.predict([[n,p,k,temp,hum,ph]]))[0]).upper())





















# import numpy as np
# distances, indices = knn.kneighbors(X)
#
# # Calculate the average distance of each data point to its k-nearest neighbors
# avg_distances = np.mean(distances, axis=1)
#
# print(X[:,0])
# print(X[:,1])
# # Plot the data points with colors based on the average distance
# plt.scatter(X[:, 0], X[:, 1], c=avg_distances, cmap='viridis')
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.title('Clustered Data (KNN)')
# plt.colorbar(label='Average Distance')
# plt.show()
# print(X[:,0])
#
# feature_indices = [0,1,2]
# selected_features = X[:, feature_indices]
# #
# #
# resolution = 0.1  # Step size for meshgrid
# x_min, x_max = selected_features[:, 0].min() - 1, selected_features[:, 0].max() + 1
# y_min, y_max = selected_features[:, 1].min() - 1, selected_features[:, 1].max() + 1
# z_min, z_max = selected_features[:, 2].min() - 1, selected_features[:, 2].max() + 1
# xx, yy, zz = np.meshgrid(np.arange(x_min, x_max, resolution),
#                          np.arange(y_min, y_max, resolution),
#                          np.arange(z_min, z_max, resolution))
#
# # Predict the labels for each point in the meshgrid
# meshgrid_points = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
# predicted_labels = knn.predict(meshgrid_points)
# predicted_labels = predicted_labels.reshape(xx.shape)
#
# # Plot the decision boundary
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(selected_features[:, 0], selected_features[:, 1], selected_features[:, 2], c=y, cmap='coolwarm')
# ax.contour3D(xx, yy, zz, predicted_labels, cmap='coolwarm', alpha=0.5)
# ax.set_xlabel('Feature 1')
# ax.set_ylabel('Feature 2')
# ax.set_zlabel('Feature 3')
# plt.title('KNN Decision Boundary')
# plt.show()