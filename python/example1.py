import mlpack
import pandas as pd
import numpy as np
# Load the dataset from an online URL.  Replace with 'covertype.csv.gz' if you
# want to use on the full dataset.
df = pd.read_csv('http://www.mlpack.org/datasets/covertype-small.csv.gz')
# Split the labels.
labels = df['label']
dataset = df.drop('label', 1)
# Split the dataset using mlpack.  The output comes back as a dictionary,
# which we'll unpack for clarity of code.
output = mlpack.preprocess_split(input=dataset,
                                 input_labels=labels,
                                 test_ratio=0.3)
training_set = output['training']
training_labels = output['training_labels']
test_set = output['test']
test_labels = output['test_labels']
# Train a random forest.
output = mlpack.random_forest(training=training_set,
                              labels=training_labels,
                              print_training_accuracy=True,
                              num_trees=10,
                              minimum_leaf_size=3)
random_forest = output['output_model']
# Predict the labels of the test points.
output = mlpack.random_forest(input_model=random_forest,
                              test=test_set)
# Now print the accuracy.  The 'probabilities' output could also be used
# to generate an ROC curve.
correct = np.sum(
    output['predictions'] == np.reshape(test_labels, (test_labels.shape[0],)))
print(str(correct) + ' correct out of ' + str(len(test_labels)) + ' (' +
    str(100 * float(correct) / float(len(test_labels))) + '%).')