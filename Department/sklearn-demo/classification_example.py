#import some libraries
import sklearn
from sklearn.ensemble import RandomForestClassifier as rf
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
#load dataset
wine_data = datasets.load_wine()
#check out the components of the dataset
print(wine_data.keys())
'''
 this shows us that we have the following keys:
`'data', 'target', 'target_names', 'DESCR', and 'feature_names'`
 data is a matrix with the feature values
 
 target is a numeric representation of the class labels
 This is something that is ubiquitous in python-based machine learning
 libraries: string class labels have to be mapped onto integer values
 
 The mapping of these features is what is contained under the target_names key
 For this dataset the string names are pretty  useless but in other instances
 they have meaningful names.
 
 'DESCR' contains a readme for the dataset
  
 'feature_names' contains information about what the columns in 'data' correspond to
'''

#Let's inspect what those features correspond to 
print(wine_data['feature_names'])

'''
Let's mix up the dataset, currently it's ordered
The shuffle function which we imported above takes multiple arrays
and scrambles them. However, it is important that the elements in the arrays
still match up. The data point in index 1 of 'data' may switch to index 60 but 
the label for data point 1 will also switch to index 60
'''
X, Y = shuffle(wine_data["data"], wine_data["target"])
#Now that our data is mixed up, we can slice it to get test and train datasets
train_X, train_Y = X[:150], Y[:150]
test_X, test_Y = X[150:], Y[150:]

#Let's initialize a random forest classifier setting some hyperparameters
model = rf(n_estimators = 25, criterion = "gini", max_depth=3)
#Now we have an initialized model but no training has been conducted
#We need to call `fit()` so that the parameters for the model are set
#in the case of our random forest classifier, this creates the little trees (stumps)
model.fit(train_X, train_Y)
#Now we have a full model! Let's predict on the test dataset
preds = model.predict(test_X)
#Let's print some results to see how we did!
print(classification_report(test_Y, preds, digits=6))
#Now go through and change this script. Change it to use an xgboost classifier
#or an svm, modify the hyperparameters used to initialize the model.
#Do you see changes in performance.
