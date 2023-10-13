# An ML Model for Automating Case Approval


## Objectives and Summary
The objective of this project was to prove out the concept of case attribute automated assignment based on other case attributes, to assess the potential value within case management applications. A publicly available data set of mortgage loan approvals was used, with the approval status (approved or declined) being the case attribute status we are aiming to predict. The following goals of the model are suggested to simulate a real scenario in which such a model might be used:
- Any approved cases will go through numerous other case management steps including additional reviews. Therefore, false positives (automated approvals where the application should have been accepted) are not unrecoverable, but do impose a business cost, as the case progresses further than it should do before the wrong decision is identified, utlising unnecessary time for additional human review.
- We assume that any cases that are rejected will be manually reviewed anyway, on the basis that no case rejection decision should be completely automated in the interest of a fair application process. Therefore, any wrongly rejected cases (false negatives) are quickly identified, at a lower cost than false positives.
- As such, we'll assume that our goal is maximise specificity (true negatives / (true negatives + false positves)) since false positives have the greatest cost, and as a secondary goal, maximise sensitivity (true positives / (true positives + false negatives), as this will improve the value added by the machine learning model.

Models considered to perform this task were k-nearest neighbor, logistic regression, decision tree, random forest, SVM and neural networks. After comparing performance on test data, the neural network was selected. Once hyper-parameters were selected, the model was re-trained and and final testing gave the following results:
- Misclassification rate: 36.7%
- Specificity: 63.3%
- Sensitivity: 63.3%

The notebook attached presents only the final model, and does not include the various models build for comparison, or the hyper parameter tuning, although these methods are discussed in the following note.

## Data
Public data source: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval/code

The dataset is provided by Analytics Vidhya on behalf of Dream Housing Finance.

The 'Home Loan Approval' dataset was chosen for the proof of concept, as the instances closely resemble a typical case record dataset subject to a case review.

A full datasheet has been seperately provided.

## Model
A summary of the model you’re using and why you chose it. 
The model selected was a Neural Network with the following arcitecture:
Net(
  (layer1): Linear(in_features=12, out_features=80, bias=True)
  (act1): ReLU()
  (layer2): Linear(in_features=80, out_features=80, bias=True)
  (act2): ReLU()
  (layer3): Linear(in_features=80, out_features=80, bias=True)
  (act3): ReLU()
  (output): Linear(in_features=80, out_features=1, bias=True)
  (sigmoid): Sigmoid()
)

The model was selected after comparing various suitable machine learning methods that had been subject to hyper-parameter optimisation and comparing performance metrics, from which the best performing model was selected. The following table summarises the methods tested and the hyperparameter values that were tuned. In order to tune parameters, a grid-search approach was taken, with heuritcally defined upper limits, lower limits and grid search spaces.




## Hyperparameters
Description of which hyperparameters you have and how you chose to optimise them. 

## Results
A summary of your results and what you can learn from your model 

You can include images of plots using the code below:
![Screenshot](image.png)

