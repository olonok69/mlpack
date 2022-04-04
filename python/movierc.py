import mlpack
import pandas as pd
import numpy as np
# First, load the MovieLens dataset.  This is taken from files.grouplens.org/
# but reposted on mlpack.org as unpacked and slightly preprocessed data.
ratings = pd.read_csv('http://www.mlpack.org/datasets/ml-20m/ratings-only.csv.gz')
movies = pd.read_csv('http://www.mlpack.org/datasets/ml-20m/movies.csv.gz')
# Hold out 10% of the dataset into a test set so we can evaluate performance.
output = mlpack.preprocess_split(input=ratings, test_ratio=0.1, verbose=True)
ratings_train = output['training']
ratings_test = output['test']
# Train the model.  Change the rank to increase/decrease the complexity of the
# model.
output = mlpack.cf(training=ratings_train,
                   test=ratings_test,
                   rank=10,
                   verbose=True,
                   algorithm='RegSVD')
cf_model = output['output_model']
# Now query the 5 top movies for user 1.
output = mlpack.cf(input_model=cf_model,
                   query=[[1]],
                   recommendations=10,
                   verbose=True)
# Get the names of the movies for user 1.
print("Recommendations for user 1:")
for i in range(10):
  print("  " + str(i) + ": " + str(movies.loc[movies['movieId'] ==
      output['output'][0, i]].iloc[0]['title']))