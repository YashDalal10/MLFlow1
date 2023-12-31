from random import Random
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

mlflow.autolog()

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)
# rfr = RandomForestRegressor(n_estimators=100, max_depth=6, max_features=3)
# rfr.fit(X_train, y_train)
# preds = rfr.predict(X_test)
model = mlflow.sklearn.load_model("runs:/23da4ea5333848f4a0c42bc6b5151908/model")
preds = model.predict(X_test)
print(preds)