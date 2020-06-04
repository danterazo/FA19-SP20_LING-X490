#This is the same as classification_example.py for most of this
import sklearn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
#load dataset
wine_data = datasets.load_wine()

X, Y = shuffle(wine_data["data"], wine_data["target"])
train_X, train_Y = X[:150], Y[:150]
test_X, test_Y = X[150:], Y[150:]

"""
Previously, we initalized a random forest classifier with some parameters
model = rf(n_estimators = 25, criterion = "gini", max_depth=3)
Wouldn't it be great if we could try out some parameters automatically?

We can actually!
"""
#First, you make an intialized model object
rf_model = rf(n_jobs=1)
"""
 We could very well just call fit on this and use the default settings. 
 However, this is generally a very bad idea. We can't really compare untuned classifiers
 Let's write some parameters that we can tune. 
"""
params = {'n_estimators':[10, 20, 30, 100],
          'criterion':['gini','entropy']}
"""
The parameters that can be tuned are specific to each model type
e.g. the options for random forest tuning are different from the ones for svm tuning
In the following line, we create a new model using our existing model, the list of parameters
and n_jobs which just specifies how many runs should be done in parallel
cv specifies how many folds to do
"""
model = GridSearchCV(rf_model, params, n_jobs=4, cv=5) 
"""
Now we have an initialized model but no training has been conducted
We need to call `fit()` 
When we call `fit` the dataset is split into `cv` pieces
For all possible combinations of parameters,
it tries to train on four of the five pieces and then tests on the fifth
"""
model.fit(train_X, train_Y)
"""
At the end of the `fit` the best parameters are used to fit over the entire
train dataset
Now we have a full model! Let's predict on the test dataset
"""
preds = model.predict(test_X)
#Let's print some results to see how we did!
print(classification_report(test_Y, preds, digits=6))
"""
Look at the sklearn documentation and figure out how to print out
The best parameters found during tuning. 
"""

"""
Now go through and change this script. 
Change it to use an xgboost classifier or an svm.
Do you see changes in performance? 
What hyperparameters did you have to change to work with 
the different classifier?

