import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import pyplot
from pandas import read_csv
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Завантаження Iris dataset
iris_dataset = load_iris()
print(f'Ключі iris_dataset: {iris_dataset.keys()}')
print(iris_dataset['DESCR'][:193] + "\n....")
print(f"Назви відповідей: {iris_dataset['target_names']}")
print(f"Назва ознак: {iris_dataset['feature_names']}")
print(f"Тип масиву data: {type(iris_dataset['data'])}")
print(f"Форма масиву data: {iris_dataset['data'].shape}")
print("Відповіді:\n{}".format(iris_dataset['target']))

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Зріз даних head
print(dataset.head(20))

# Статистичні зведення методом describe
print(dataset.describe())

# Розподіл за атрибутом class
print(dataset.groupby('class').size())

# Діаграма розмаху
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()

# Гістограма розподілу атрибутів датасета
dataset.hist()
pyplot.show()

# Матриця діаграм розсіювання
sns.pairplot(dataset)
# Заміна pyplot.show() на sns.plt.show()
plt.show()

# Розділення датасету на навчальну та контрольну вибірки
array = dataset.values

# Вибір перших 4-х стовпців
X = array[:, 0:4]

# Вибір 5-го стовпця
y = array[:, 4]

X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

models = []
models.append(('LR', RandomForestClassifier()))
models.append(('LDA', GradientBoostingClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', KMeans()))

results = []
names = []

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    accuracy = accuracy_score(Y_validation, predictions)
    print('%s: %f' % (name, accuracy))
    results.append(accuracy)
    names.append(name)

# Заміна pyplot.boxplot на sns.boxplot
sns.boxplot(x=names, y=results)
pyplot.title('Algorithm Comparison')
# Заміна pyplot.show() на sns.plt.show()
pyplot.show()

# Створюємо прогноз на контрольній вибірці
model = KMeans()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)

# Конвертація строкових міток в числові
le = LabelEncoder()
Y_validation_numeric = le.fit_transform(Y_validation)
predictions_numeric = le.transform(predictions)

# Оцінюємо прогноз
print(accuracy_score(Y_validation_numeric, predictions_numeric))
print(confusion_matrix(Y_validation_numeric, predictions_numeric))
print(classification_report(Y_validation_numeric, predictions_numeric))

X_new = np.array([[5, 2.9, 1, 0.2]])

for name, model in models:
    model.fit(X_train, Y_train)
    prediction = model.predict(X_new)
    print("Прогноз: {}".format(prediction))
    print(accuracy_score(Y_validation_numeric, predictions_numeric))
    print(confusion_matrix(Y_validation_numeric, predictions_numeric))
    print(classification_report(Y_validation_numeric, predictions_numeric))