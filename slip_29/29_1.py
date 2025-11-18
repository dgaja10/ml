from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
iris = datasets.load_iris()
X = iris.data  
y = iris.target  
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.3, random_state=42)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy after PCA and SVM classification: {accuracy:.2f}")
new_sample = [[5.1, 3.5, 1.4, 0.2]]  
new_sample_reduced = pca.transform(new_sample)
predicted_class = model.predict(new_sample_reduced)
predicted_flower = iris.target_names[predicted_class[0]]
print(f"The predicted flower type for the new measurements is: {predicted_flower}")
