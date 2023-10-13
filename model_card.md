# Model Card

See the [example Google model cards](https://modelcards.withgoogle.com/model-reports) for inspiration. 

## Model Description

**Input:** There are 12 inputs to the model: applicant gender, applicant marital status, applicant's number of dependents, applicants graduate status, applicants self-employed status, applicant's income, co-applicants income, loan amount, loan term, if the applicants credit history is available, the property area.

**Output:** The output of the model is a binary prediction of if the applicants home loan application will be approved or rejected.

**Model Architecture:** The model is a Neural network, with 12 inputs, 1 output, 3 hidden layers each of 80 neurons and with a ReLU activation function.

## Performance
A random 75:25 split of the 480 remaining datapoints (after filtering out cases with missing attributes) was used to test the model.
Misclassification rate: 35.0%
Specificity: 70.7%
Sensitivity: 62.0%
Where:
Misclassification rate = no. wrong classification / total data points.
Specificity = true negatives / (true negatives + false positves).
Sensitivity = true positives / (true positives + false negatives).
Note that it has been identified that false positives are most essential to minimise, so the specificity is the most important performance metric.

## Limitations

The model was trained with cases that did not contain any missing values.
The total of 480 datapoints used is a very limited dataset.
Given legislation surrounding applications of ML models, the model can not be solely used to make a decision on whether or not an individuals loan approval is approved or rejected. However, it can be used to give reviewers some insight, or to help hightlight the applications that require most attention when reviewing.

## Trade-offs

Whilst training the model, over-sampling methods were used to account for the excess of approved cases in the training set and aim to improve the specificity of the model. This was found to negatively impact the sensitivity.
