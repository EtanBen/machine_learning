# Split the data 
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.multiclass import  OneVsOneClassifier,OneVsRestClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Load processed feature matrix and labels
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
from skimage.filters import sobel
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
#matice de confusion
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import time 
from sklearn.multiclass import OneVsRestClassifier

# TODO: Add any util functions you may have from the previous script
digits = load_digits()
print(digits.data.shape)

def get_statistics_text(targets):
    # TODO: Write your code here, returning at least the following useful infos:
    # * Label names
    # * Number of elements per class
    df=pd.DataFrame({'Classe': targets})
    compter=df['Classe'].value_counts().sort_index()
    print("distribution")
    #print(compter)
    
    return compter


# TODO: Load the raw data
X= digits.data
y= digits.target

#####
#In machine learning, we must train the model on one subset of data and test it on another.
#This prevents the model from memorizing the data and instead helps it generalize to unseen examples.
#The dataset is typically divided into:
#Training set → Used for model learning.
#Testing set → Used for evaluating model accuracy.
# The training set is also split as a training set and validation set for hyper-parameter tunning. This is done later
#
# Split dataset into training & testing sets


##########################################
## Train/test split and distributions
##########################################


# 1- Split dataset into training & testing sets
# TODO: FILL OUT THE CORRECT SPLITTING HERE
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=42)  
### If you want, you could save the data, this would be a good way to test your final script in the same evaluation mode as what we will be doing
np.save("X_train.npy", X_train)
np.save("test_data.npy", X_test)
np.save("y_train.npy", y_train)
np.save("test_label.npy", y_test)

# TODO: Print dataset split summary...

print("Dataset original:")
print("Nb échantillons:",len(y))
print("Nb classes:", len(np.unique(y)))
original=get_statistics_text(y)
print(original)

print("Dataset d'entrainement:")
print("Nb échantillons:",len(y_train))
print("Nb classes:", len(np.unique(y_train)))
train=get_statistics_text(y_train)
print(train)

print("Dataset de test:")
print("Nb échantillons:",len(y_test))
print("Nb classes:", len(np.unique(y_test)))
test=get_statistics_text(y_test)
print(test)


# TODO: ... and plot graphs of the three distributions in a readable and useful manner (bar graph, either side by side, or with some transparancy)


#sns.barplot(x=a.index, y=a.values)
#plt.show()

#sns.barplot(x=b.index, y=b.values)
#plt.show()

#sns.barplot(x=c.index, y=c.values)
#plt.show()



les_3 = pd.DataFrame({'Original': original,'Train': train,'Test': test })

# Tracer les trois distributions sur le même graphique
les_3.plot(kind='bar', width=0.8)
plt.title('Distribution des classes dans les 3 datasets')
plt.xlabel('Classe')
plt.ylabel('Nb échantillons')
plt.legend(title='Dataset')
#plt.grid(True, axis='y', linestyle='--')
plt.tight_layout()
plt.show()

#train test et original


# Avec la normalisation
#normaliser que pour chaque ligne ait la même somme : df.div(df.sum(axis = 1), axis = 0)

somme=les_3.sum(axis=0)
div=les_3.div(somme, axis=1) 
normalize = div * 100

# Tracer les trois distributions normalisées sur le même graphique
ax = normalize.plot(kind='bar')
plt.title('Distribution normalisée des classes dans les 3 datasets')
plt.xlabel('Classe')
plt.ylabel('% échantillons')
plt.legend(title='Dataset')

plt.tight_layout()
plt.show()


# TODO: (once the learning has started, and to be documented in your report) - Impact: Changing test_size affects model training & evaluation.


##########################################
## Prepare preprocessing pipeline
##########################################

# We are trying to combine some global features fitted from the training set
# together with some hand-computed features.
# 
# The PCA shall not be fitted using the test set. 
# The handmade features are computed independently from the PCA
# We therefore need to concatenate the PCA computed features with the zonal and 
# edge features. 
# This is done with the FeatureUnion class of sklearn and then combining everything in
# a Pipeline.
# 
# All elements included in the FeatureUnion and Pipeline shall have at the very least a
# .fit and .transform method. 
#
# Check this documentation to understand how to work with these things 
# https://scikit-learn.org/stable/auto_examples/compose/plot_feature_union.html#sphx-glr-auto-examples-compose-plot-feature-union-py

# Example of wrapper for adding a new feature to the feature matrix
from sklearn.base import BaseEstimator, TransformerMixin

class EdgeInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute an average Sobel estimator on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self # No fitting needed for this processing
    
    def transform(self, X):
        sobel_feature = np.array([np.mean(sobel(img.reshape((8,8)))) for img in X]).reshape(-1, 1)
        return sobel_feature

# TODO: Fill out the useful code for this class


class ZonalInfoPreprocessing(BaseEstimator, TransformerMixin):
    '''A class used to compute zone information on the image
       This class can be used in conjunction of other feature engineering
       using Pipelines or FeatureUnion

       TODO: Continue this work
    '''
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        # No fitting needed
        return self
    
    def transform(self, X):
        features = []
        for image in X:
            img = image.reshape(8, 8)
            zone1 = np.mean(img[0:3, :])  
            zone2 = np.mean(img[3:5, :])  
            zone3 = np.mean(img[5:8, :])  
            features.append([zone1, zone2, zone3])
        return np.array(features)


# TODO: Create a single sklearn object handling the computation of all features in parallel
all_features = FeatureUnion([('zonal', ZonalInfoPreprocessing()),('edges', EdgeInfoPreprocessing()),('pca',PCA(n_components=28))])
F = all_features.fit(X_train,y).transform(X_train)
# Let's make sure we have the number of dimensions that we expect!
print("Nb features computed: ", F.shape[1])



# Now combine everything in a Pipeline
# The clf variable is the one which plays the role of the learning algorithms
# The Pipeline simply allows to include the data preparation step into it, to 
# avoid forgetting a scaling, or a feature, or ...
# 
# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
# TODO: Write your own pipeline, with a linear SVC classifier as the prediction
clf = Pipeline([('Standardscaler',StandardScaler()),('features', all_features),('MinMaxscaler',MinMaxScaler()),('svc',SVC(kernel='linear',random_state=42))])
# COMMENT : which pipeline modal is optimal, how to put the scalers and features 
##########################################
## Premier entrainement d'un SVC
##########################################

# TODO: Train your model via the pipeline
clf.fit(X_train,y_train)
predict_test = clf.predict(X_test)
predict_train = clf.predict(X_train)
# TODO: Predict the outcome of the learned algorithm on the train set and then on the test set 
#predict_test = [1]*len(X_test)
#predict_train = [1]*len(X_train)

print("Accuracy of the SVC on the test set: ", sum(y_test==predict_test)/len(y_test))
print("Accuracy of the SVC on the train set: ", sum(y_train==predict_train)/len(y_train))

# TODO: Look at confusion matrices from sklearn.metrics and 
# 1. Display a print of it
# 2. Display a nice figure of it
# 3. Report on how you understand the results


# Matrice de confusion pour test 
cm1 = confusion_matrix(y_test, predict_test, labels=clf.classes_)
disp1= ConfusionMatrixDisplay(confusion_matrix=cm1, display_labels=clf.classes_)
disp1.plot()
plt.title('Matrice de confusion pour le Test set')
plt.show()

# Matrice de confusion pour train 
cm2= confusion_matrix(y_train, predict_train, labels=clf.classes_)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2, display_labels=clf.classes_)
disp2.plot()
plt.title('Matrice de confusion pour le train set')
plt.show()

# Matrice de confusion sur toutes les données (train et test)
X_tout = np.vstack((X_train, X_test))
y_tout = np.concatenate((y_train, y_test))
y_pred_tout = clf.predict(X_tout)

cm3 = confusion_matrix(y_tout, y_pred_tout)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3, display_labels=clf.classes_)
disp3.plot()
plt.title('Matrice de confusion avec toutes le données')
plt.show()



# TODO: Work out the following questions (you may also use the score function from the classifier)
print("\n Question: How does changing test_size influence accuracy?")
print("Try different values like 0.1, 0.3, etc., and compare results.\n")


##########################################
## Hyper parameter tuning and CV
##########################################
# TODO: Change from the linear classifier to an rbf kernel
# TODO: List all interesting parameters you may want to adapt from your preprocessing and algorithm pipeline
# TODO: Create a dictionary with all the parameters to be adapted and the ranges to be tested
clf_rbf = Pipeline([('features', all_features),('scaler', StandardScaler()),('svc', SVC(kernel='rbf'))])


param_grid = {'features__pca__n_components': [10, 20, 30, 40], 'classifier__c':[0.1, 0.5, 10, 50],'classifier__gamma': [0.01, 0.1, 1, 10]}
#param_grid = {'features__pca__n_components': [10, 20, 30, 40],  'classifier__kernel': ['linear', 'rbf'],'classifier__c':[0.1, 0.5, 10, 50],'classifier__gamma': [0.01, 0.1, 1, 10]}

#parametre gamma de rbf à comprendre 
#DO NOT use kernel linear with gamma classifier -->> either do one and leave the other or use mask (check documentation)

# TODO: Use a GridSearchCV on 5 folds to optimize the hyper parameters
from sklearn.model_selection import GridSearchCV

grid = GridSearchCV(clf_rbf, param_grid, cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)


# TODO: fit the grid search CV and 
# 1. Check the results
print("Best parameters:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)
# 2. Update the original pipeline (or create a new one) with all the optimized hyper parameters
# 3. Retrain on the whole train set, and evaluate on the test set
#update the original pipeline with the best parameters
best_params = grid.best_params_
clf_rbf.set_params(**best_params)
clf_rbf.fit(X_train, y_train)
y_pred = clf_rbf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on test set:", accuracy)

# 4. Answer the questions below and report on your findings
from sklearn.model_selection import GridSearchCV

print("\nQuestion: What happens if we change K from 5 to 10?\n")

for k in [3, 5, 10]:
    print(f"\nRunning GridSearchCV with K={k}-fold cross-validation:")
    
    grid_k = GridSearchCV(
        clf_rbf,
        param_grid={
            'features__pca__n_components': [10, 20, 30],
            'svc__C': [0.1, 1, 10],
            'svc__gamma': [0.01, 0.1, 1]
        },
        cv=k,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0
    )
    
    grid_k.fit(X_train, y_train)
    print(f"- Best CV score with K={k}: {grid_k.best_score_:.4f}")
    print(f"- Best parameters with K={k}: {grid_k.best_params_}")
    
#print(" K-Fold Cross-Validation Results:")
#print(f"- Best Cross-validation score: {grid_search.best_score_}")
#print(f"- Best parameters found: {grid_search.best_estimator_}")
#####
#print("\n Question: What happens if we change K from 5 to 10?")
#print("Test different K values and compare the accuracy variation.\n")


##########################################
## OvO and OvR
##########################################
# TODO: Using the best found classifier, analyse the impact of one vs one versus one vs all strategies
# Analyse in terms of time performance and accuracy


# Print OvO results
print(" One-vs-One (OvO) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Suitable for small datasets but increases complexity.")

print("\n Question: How does OvO compare to OvR in execution time?")
print("Try timing both methods and analyzing efficiency.\n")
###################
# TODO:  One-vs-Rest (OvR) Classification


# Print OvR results
print(" One-vs-Rest (OvR) Classification:")
print(f"- Test score: {clf.score(X_test, y_test)}")
print(f"- Number of classifiers trained: {len(clf.get_params('classifier__estimators_'))}")
print("- Impact: Better for large datasets but less optimal for highly imbalanced data.")

print("\n Question: When would OvR be better than OvO?")
print("Analyze different datasets and choose the best approach!\n")
########


#meilleur taille de test pour avoir la meilleure exactitude 
def evaluate_model(test_size):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def plot_test_size_effect():
    test_sizes = np.arange(0.1, 0.9, 0.1)
    accuracies = []
    for size in test_sizes:
        accuracy= evaluate_model(size)
        accuracies.append(accuracy)
    plt.plot(test_sizes, accuracies, marker='o')
    plt.title('Exactitude en fonction de la taille du test')
    plt.xlabel('Taille du test')
    plt.ylabel('Exactitude')
    plt.tight_layout()
    plt.show()
plot_test_size_effect()



def tmp_ovr(X_train, y_train, X_test, y_test):
    all_features = FeatureUnion([
        ('zonal', ZonalInfoPreprocessing()),
        ('edges', EdgeInfoPreprocessing()),
        ('pca', PCA(n_components=28))
    ])

    clf_ovr = Pipeline([
        ('standard', StandardScaler()),
        ('features', all_features),
        ('minmax', MinMaxScaler()),
        ('svc', OneVsRestClassifier(SVC(kernel='linear', random_state=42)))
    ])

    start = time.time()
    clf_ovr.fit(X_train, y_train)
    end = time.time()

    acc_train = clf_ovr.score(X_train, y_train)
    acc_test = clf_ovr.score(X_test, y_test)
    duration = end - start

    print("OVR :")
    print(f"  Accuracy train : {acc_train:.4f}")
    print(f"  Accuracy test  : {acc_test:.4f}")
    print(f"  Temps d'entraînement : {duration:.4f} s\n")

    return clf_ovr, duration
tmp_ovr(X_train, y_train, X_test, y_test)





def tmp_ovo(X_train, y_train, X_test, y_test):
    all_features = FeatureUnion([
        ('zonal', ZonalInfoPreprocessing()),
        ('edges', EdgeInfoPreprocessing()),
        ('pca', PCA(n_components=28))
    ])

    clf_ovo = Pipeline([
        ('standard', StandardScaler()),
        ('features', all_features),
        ('minmax', MinMaxScaler()),
        ('svc', OneVsOneClassifier(SVC(kernel='linear', random_state=42)))
    ])

    start = time.time()
    clf_ovo.fit(X_train, y_train)
    end = time.time()

    acc_train = clf_ovo.score(X_train, y_train)
    acc_test = clf_ovo.score(X_test, y_test)
    duration = end - start

    print("OVO :")
    print(f"  Accuracy train : {acc_train:.4f}")
    print(f"  Accuracy test  : {acc_test:.4f}")
    print(f"  Temps d'entraînement : {duration:.4f} s\n")

    return clf_ovo, duration
tmp_ovo(X_train, y_train, X_test, y_test)

