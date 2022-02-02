using Statistics: median

"""
    KNNDetector(k=5,
                metric=Euclidean,
                algorithm=:kdtree,
                leafsize=10,
                reorder=true,
                reduction=:maximum)

Calculate the anomaly score of an instance based on the distance to its k-nearest neighbors.

Parameters
----------
$K_PARAM

$KNN_PARAMS

    reduction::Symbol
One of `(:maximum, :median, :mean)`. (`reduction=:maximum`) was proposed by [1]. Angiulli et al. [2] proposed sum to
reduce the distances, but mean has been implemented for numerical stability.

Examples
--------
$(SCORE_UNSUPERVISED("KNNDetector"))

References
----------
[1] Ramaswamy, Sridhar; Rastogi, Rajeev; Shim, Kyuseok (2000): Efficient Algorithms for Mining Outliers from Large Data
Sets.

[2] Angiulli, Fabrizio; Pizzuti, Clara (2002): Fast Outlier Detection in High Dimensional Spaces.
"""
OD.@detector mutable struct KNNDetector <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree, :brutetree))
    static::Union{Bool,Symbol} = :auto::(_ in (true, false, :auto))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    reduction::Symbol = :maximum::(_ in (:maximum, :median, :mean))
end

struct KNNModel <: DetectorModel
    tree::NN.NNTree
end

function OD.fit(detector::KNNDetector, X::Data; verbosity)::Fit
    X = prepare_data(X, detector.static)

    # create the specified tree
    tree = @tree detector X

    # use tree to calculate distances
    _, dists = detector.parallel ?
               knn_parallel(tree, X, detector.k, true) :
               knn_sequential(tree, X, detector.k, true)

    # reduce distances to outlier score
    scores = _knn(dists, detector.reduction)
    return KNNModel(tree), scores
end

function OD.transform(detector::KNNDetector, model::KNNModel, X::Data)::Scores
    X = prepare_data(X, detector.static)
    _, dists = detector.parallel ?
               knn_parallel(model.tree, X, detector.k) :
               knn_sequential(model.tree, X, detector.k)
    return _knn(dists, detector.reduction)
end

@inline function _knn(distances::AbstractVector{<:AbstractVector}, reduction::Symbol)::Scores
    # Helper function to reduce `k` distances to a single distance.
    if reduction == :maximum
        return maximum.(distances)
    elseif reduction == :median
        return median.(distances)
    elseif reduction == :mean
        return mean.(distances)
    end
end
