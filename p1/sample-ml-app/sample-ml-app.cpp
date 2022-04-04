#include "mlpack/core.hpp"
#include "mlpack/methods/random_forest/random_forest.hpp"
#include "mlpack/methods/decision_tree/random_dimension_select.hpp"
#include "mlpack/core/cv/k_fold_cv.hpp"
#include "mlpack/core/cv/metrics/accuracy.hpp"
#include "mlpack/core/cv/metrics/precision.hpp"
#include "mlpack/core/cv/metrics/recall.hpp"
#include "mlpack/core/cv/metrics/F1.hpp"


using namespace arma;
using namespace mlpack;
using namespace mlpack::tree;
using namespace mlpack::cv;