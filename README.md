### Support Vector Machine ###
-------------
Made from scratch two class support vector machine optimizer and tester.

### Dependencies ###
numpy - install with "pip install numpy"

pandas - install with "pip install pandas"
### Usage ###
To create a classifier for your own data run the following command with any or all of the optional arguments shown below.

python3 app.py -f \<csv file path> -c \<start index for feature columns>  \<end index for feature columns> \<index of classification column> -r \<starting row index of training data> \<ending row index of training data> -e \<starting row index of test data> \<ending row index of test data> -l \<start index of unclassified data> \<end index of unclassified data> -t \<learning rate> -i \<number of iterations to train model>

To test out the classifier on a set of star data run the following command. This program tries to predict whether or not a star is a pulsar or not using the same algorithm from before.

python3 pulsar_stars_classification.py

