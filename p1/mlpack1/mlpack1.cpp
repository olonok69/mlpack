// This is an interactive demo, so feel free to change the code and click the 'Run' button.

// This simple program uses the mlpack::neighbor::NeighborSearch object
// to find the nearest neighbor of each point in a dataset using the L1 metric,
// and then print the index of the neighbor and the distance of it to stdout.

#include <mlpack/core.hpp>
#include <mlpack/methods/neighbor_search/neighbor_search.hpp>

#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"

using namespace mlpack;
using namespace mlpack::neighbor; // NeighborSearch and NearestNeighborSort
using namespace mlpack::metric; // ManhattanDistance
using namespace arma;
using namespace mlpack::tree;
using namespace mlpack::cv;


int main()
{
    mat dataset;
    bool loaded = mlpack::data::Load("data/german.csv", dataset);
    if (!loaded)
        return -1;

    Row<size_t> labels;
    labels = conv_to<Row<size_t>>::from(dataset.row(dataset.n_rows - 1));
    dataset.shed_row(dataset.n_rows - 1);

    const size_t numClasses = 2;
    const size_t minimumLeafSize = 5;
    const size_t numTrees = 10;
    RandomForest<GiniGain, RandomDimensionSelect> rf;
    rf = RandomForest<GiniGain, RandomDimensionSelect>(dataset, labels,
        numClasses, numTrees, minimumLeafSize);

    Row<size_t> predictions;
    rf.Classify(dataset, predictions);
    const size_t correct = arma::accu(predictions == labels);
    cout << "\nTraining Accuracy: " << (double(correct) / double(labels.n_elem));


    return 0;
}

