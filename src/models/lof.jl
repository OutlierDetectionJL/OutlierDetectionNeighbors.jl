using Statistics: mean

"""
    LOFDetector(k = 5,
                metric = Euclidean(),
                algorithm = :kdtree,
                leafsize = 10,
                reorder = true,
                parallel = false)

Calculate an anomaly score based on the density of an instance in comparison to its neighbors. This algorithm introduced
the notion of local outliers and was developed by Breunig et al., see [1].

Parameters
----------
$K_PARAM

$KNN_PARAMS

Examples
--------
$(SCORE_UNSUPERVISED("LOFDetector"))

References
----------
[1] Breunig, Markus M.; Kriegel, Hans-Peter; Ng, Raymond T.; Sander, Jörg (2000): LOF: Identifying Density-Based Local
Outliers.
"""
OD.@detector mutable struct LOFDetector <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree, :brutetree))
    static::Union{Bool,Symbol} = :auto::(_ in (true, false, :auto))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
end

struct LOFModel <: DetectorModel
    # For efficient prediction, we need to store the learned tree, the distances of each training sample to its
    # k-nearest neighbors, as well as the training lrds.
    tree::NN.NNTree
    ndists::AbstractMatrix
    lrds::AbstractVector
end

function OD.fit(detector::LOFDetector, X::Data; verbosity)::Fit
    X = prepare_data(X, detector.static)

    # create the specified tree
    tree = @tree detector X

    # use tree to calculate distances
    idxs, dists = detector.parallel ?
                  knn_parallel(tree, X, detector.k, true) :
                  knn_sequential(tree, X, detector.k, true)

    # transform dists (vec of vec) to matrix to allow faster indexing later
    ndists = reduce(hcat, dists)

    # pre calculate lrds for later prediction use
    lrds = _calculate_lrd(idxs, dists, ndists, detector.k)

    # reduce distances to outlier score
    scores = _lof_from_lrd(idxs, lrds)

    return LOFModel(tree, ndists, lrds), scores
end

function OD.transform(detector::LOFDetector, model::LOFModel, X::Data)::Scores
    X = prepare_data(X, detector.static)
    idxs, dists = detector.parallel ?
                  knn_parallel(model.tree, X, detector.k, false, true) :
                  knn_sequential(model.tree, X, detector.k, false, true)
    return _lof(idxs, dists, model.ndists, model.lrds, detector.k)
end

function _lof(idxs::AbstractVector, dists::AbstractVector, model_dists::AbstractArray,
    model_lrds::AbstractVector, k::Int)::Scores
    lrds = _calculate_lrd(idxs, dists, model_dists, k)
    # calculate the local outlier factor from the lrds
    map((idx, lrd) -> mean(model_lrds[idx]) / lrd, idxs, lrds)
end

function _lof_from_lrd(idxs::AbstractVector, lrds::AbstractVector)::Scores
    # Directly calculate the local outlier factor from idxs with corresponding lrds.
    map((idx, lrd) -> mean(lrds[idx]) / lrd, idxs, lrds)
end

function _calculate_lrd(idxs::AbstractVector, dists::AbstractVector, ndists::AbstractArray, k::Int)::AbstractVector
    # The LRD of a sample is the inverse of the average reachability distance of its k-nearest neighbors. Epsilon is
    # added in case that there are more than k duplicates.
    map((is, ds) -> 1 / (mean(_max!(ndists[k, is], ds)) + 1e-10), idxs, dists)
end

function _max!(ar1::AbstractVector, ar2::AbstractVector)::AbstractVector
    # Calculate the element wise maximum between two vectors.
    for i in eachindex(ar1)
        @inbounds ar1[i] = ar1[i] > ar2[i] ? ar1[i] : ar2[i]
    end
    ar1
end
