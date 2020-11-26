import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv(r"C:\Users\Mahesh Sharma\Desktop\HealthApp\data\cancer.csv")
df.drop(df.columns[[0,-1]], axis=1, inplace=True)
# Split the features data and the target 
Xdata = df.drop(['diagnosis'], axis=1)
ydata = df['diagnosis']

# Encoding the target value 
yenc = np.asarray([1 if c == 'M' else 0 for c in ydata])
cols = ['concave points_mean','area_mean','radius_mean','perimeter_mean','concavity_mean',]
Xdata = df[cols]
print(Xdata.columns)

X_train, X_test, y_train, y_test = train_test_split(Xdata, yenc, 
                                                    test_size=0.3,
                                                    random_state=43)
print('Shape training set: X:{}, y:{}'.format(X_train.shape, y_train.shape))
print('Shape test set: X:{}, y:{}'.format(X_test.shape, y_test.shape))

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

clf_report = classification_report(y_test, y_pred)
print('Classification report')
print("---------------------")
print(clf_report)
print("_____________________")

joblib.dump(model,r"C:\Users\Mahesh Sharma\Desktop\HealthApp\Indivisual_Deployment\Breast\cancer_model.pkl")
