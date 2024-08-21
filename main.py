import streamlit as st
import seaborn as sns
df = sns.load_dataset('iris')
df

x = df.iloc[:, :-1]
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

st.sidebar.title('Classification')

classifier = st.sidebar.selectbox('Classifier', ('KNN', 'Decision Tree', 'Random Forest', 'SVM'))

from sklearn.metrics import accuracy_score
y_pred = knn.predict(x_test)
accuracy_score(y_test, y_pred)

from sklearn.svm import SVC
svm = SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)
accuracy_score(y_test, y_pred)
