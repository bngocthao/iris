# Nạp các gói thư viện cần thiết
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Đọc tập tin json chứa tập dữ liệu iris
iris =pd.read_json('https://raw.githubusercontent.com/ltdaovn/dataset/master/iris.json')
print('Dataset info:\n', iris.info)
X = iris.drop(columns=['species'])
y = iris.species
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
# #print(y_pred)
# # Tính độ chính xác
# print("Do chinh xac cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" %
accuracy_score(y_test, y_pred)
# # Hiển thị cây
# tree.plot_tree(model.fit(X, y))
# plt.show()


def pred(a, b, c, d):
  pre = model.predict([[a, b, c, d]])
  return pre


import pickle
output = open('data1.pkl', 'wb')
pickle.dump(model, output)

