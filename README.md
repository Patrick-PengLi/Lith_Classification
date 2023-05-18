# BestSubset_RandomForest
A workflow for lithology classification from well logs. 

The proposed approach consists of two steps: a feature selection step and a machine learning classification step. 

The best subset regression is used for feature selection to investigate the correlation between input well logs and target lithologies based on the core data analysis. Then a random forest classifier is implemented for lithology classification. 

The novelty of this workflow is the feature selection process, i.e., best subset regression, that aims to obtain the optimal accuracy score. The implementation of feature selection reduces the number of variables used in the classification, and consequently reduces the prediction bias of the model and improves the computational efficiency of the classifier.

The feature selection method and classification method can be replaced by any other methods

 
![image](https://github.com/Patrick-PengLi/BestSubset_RandomForest/assets/70155353/cb21fbbf-094b-4f88-822b-3245372f8f71)
 

![image](https://github.com/Patrick-PengLi/BestSubset_RandomForest/assets/70155353/a053d093-db75-4100-9bf6-99c4edc294db)

