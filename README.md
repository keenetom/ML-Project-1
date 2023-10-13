# An ML Model for Automating Case Approval


## Objectives and Summary
The objective of this project was to prove out the concept of case attribute automated assignment based on other case attributes, to assess the potential value within case management applications. A publicly available data set of mortgage loan approvals was used, with the approval status (approved or declined) being the case attribute status we are aiming to predict. The following goals of the model are suggested to simulate a real scenario in which such a model might be used:
- Any approved cases will go through numerous other case management steps including additional reviews. Therefore, false positives (automated approvals where the application should have been accepted) are not unrecoverable, but do impose a business cost, as the case progresses further than it should do before the wrong decision is identified, utlising unnecessary time for additional human review.
- We assume that any cases that are rejected will be manually reviewed anyway, on the basis that no case rejection decision should be completely automated in the interest of a fair application process. Therefore, any wrongly rejected cases (false negatives) are quickly identified, at a lower cost than false positives.
- As such, we'll assume that our goal is maximise specificity (true negatives / (true negatives + false positves)) since false positives have the greatest cost, and as a secondary goal, maximise sensitivity (true positives / (true positives + false negatives), as this will improve the value added by the machine learning model.

Models considered to perform this task were k-nearest neighbor, logistic regression, decision tree, random forest, SVM and neural networks. After comparing performance on test data, the neural network was selected. Once hyper-parameters were selected, the model was re-trained and and final testing gave the following results:
- Misclassification rate: 35.0%
- Specificity: 70.7%
- Sensitivity: 62.0%

The notebook attached presents only the final model, and does not include the various models build for comparison, or the hyper parameter tuning, although these methods are discussed in the following note.

## Data
Public data source: https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval/code

The dataset is provided by Analytics Vidhya on behalf of Dream Housing Finance.

The 'Home Loan Approval' dataset was chosen for the proof of concept, as the instances closely resemble a typical case record dataset subject to a case review.

A full datasheet has been seperately provided.

## Model
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

The model was selected after comparing various suitable machine learning methods that had been subject to hyper-parameter optimisation and comparing performance metrics, from which the best performing model was selected. The table found in the image file 'Approach comparison.png' summarises the methods tested and the hyperparameter values that were tuned. In order to tune parameters, a grid-search approach was taken, with heuristically defined upper limits, lower limits and grid search spaces.

## Hyperparameters
In order to reduce the search space of hyper-parameter optimisation, the following limits were imposed to simplify the model:
- Assume all hidden layers will have the same number of neurons
- Assume all hidden layers will have the same activation function
Hyper-parameters were optimised through a 2-step grid search approach.
Step 1: 7 parameters were given two possible options as follows. These were heuristically chosen, and considered upper and lower bounds. With more time, a more thorough approach to justify the full search space for hyperparamters should be taken.
- Number of Epochs: 100 or 1000
- Training Batch Size: 30 or 150
- Activation Function: ReLU or Sigmoid
- Learning rate: 0.1 or 0.001
- Momentum: 0.5 or 0.99
- No. Layers: 2 or 5
- No. Neurons per layer: 20 or 100
A grid search of all possible combinations was performed, from which it was observed that all the best performing models had 1000 epochs, the ReLU activation function and a training batch size of 30. These values were set, and for the remaining hyper-parameters for which the performance dependency was not clear, another grid search approach was taken:
- Learning rate: 0.005, 0.01, 0.05
- Momentum: 0.5, 0.75, 0.99
- Number of layers: 2,3,4
- No. Neurons per layer: 60, 80, 100
A grid search was performed, and the best performing model was selected.
Note that all other methods were also subject to hyper parameter optimisation, however typically only required 1 or 2 hyper parameters to be optimised, which was also achieved through a grid search approach.

## Results
The final performance metric of the model presented are:
- Misclassification rate: 35.0%
- Specificity: 70.7%
- Sensitivity: 62.0%

Whilst the model can provide some good insight into whether or not a case is likely to be rejected / approved and can potentially offer some value in advising the reviewer, it is clear that a case management solution could not rely entirely on an ML model to make a decision based on this particular use case. However, the study does prove that there is potential value in using ML models to assign attributes to cases far more effectively than a naive approach.



