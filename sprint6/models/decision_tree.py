import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('/datasets/train_data.csv')

df.loc[df['last_price'] > 5650000, 'price_class'] = 1
df.loc[df['last_price'] <= 5650000, 'price_class'] = 0

features = df.drop(['last_price', 'price_class'], axis=1)
target = df['price_class']

model = DecisionTreeClassifier(random_state=12345)

model.fit(features, target)

train_predictions = model.predict(features)

test_df = pd.read_csv('/datasets/test_data.csv')

test_df.loc[test_df['last_price'] > 5650000, 'price_class'] = 1
test_df.loc[test_df['last_price'] <= 5650000, 'price_class'] = 0

test_features = test_df.drop(['last_price', 'price_class'], axis=1)
test_target = test_df['price_class']
test_predictions = model.predict(test_features)

accuracy = accuracy_score(target, train_predictions)
test_accuracy = accuracy_score(test_target, test_predictions)

print("Accuracy")
print("Обучающая выборка:", accuracy)
print("Тестовая выборка:", test_accuracy)