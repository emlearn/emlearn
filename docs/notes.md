
# Documentation strategy

Kinds of documentation

- Getting Started. Highest priority. Shows how to achieve simple task in 10 minutes. Very easy steps to follow.
- Application examples. High priority. Show/explain what can / has been achieved with project. 
- References. API etc. Should be enough to search and find info when doing a bit different things.
- Internal docs. Optimizations we implement. Design choices.

## Target audiences

- Researchers in embedded Machine Learning.
Making proof-of-concepts / feasibility studies etc
- Embedded engineers integrating ML into products
- Machine Learning engineers wanting to deploy on microcontroller/embedded
- Intermediate hobbyists/makers with practical skills in ML/embedded/programming/electronics

Non-target:

- Absolute beginners to MachineLearning/embedded/electronics/programming.


## Prerequisite knowledge

Assumed background knowledge

- Machine Learning basics. Learning setups, models, evaluation.
- Data Science basics. Wrangling datasets, visualizing
- Digital Signal Processing basics. Time-series
- Practical programming skills. In Python and C/C++
- Practical embedded skills. Programming MCU, communicating with PC, debugging
- Practical electronic skills.

Not assumed.

- Optimizing ML model size/costs


# Learning goals

Basics

- Convert a model to C, run on computer
- Run model on microcontroller. E.g. Arduino
- Optimize model. Hyperparameters, conversion parameters. Ex RandomForest
- Collect data from microcontroller. Build model on this data

Classification

- Classifier-as-detector for time-series signals
- Time-conditional modeling

Anomaly detection

- Static anomaly detection
- Anomaly detection conditional on time or context
- Combined with classifier, unknown class

TinyML benefits

- Reacting quickly. Stop system on issue.
- Save power. Only transmit data that is needed
- Standalone operation. User interface on device

ML benefits

- Handle complex phenomena.
Multi-variate. Multi-modal. Complex variable interactions. Complex time dependency.
- Data-driven development.
Structured methodology
Known cases covered. Known performance.

Misc

- Feature engineering?
Normalization
Power-transforms / box-cox

# Usage goals
What peoples should be able to do.

- Get an initial model working
Without spending a lot of optimizing performance

- Find a model that fits within compute constraints C,
while reaching performance target T

- Find the Pareto optimal model
Accuracy vs compute/energy

- Collect a dataset
Including labeling


