"""
    COFDetector(k = 5,
                metric = Euclidean(),
                algorithm = :kdtree,
                leafsize = 10,
                reorder = true,
                parallel = false)

Local outlier density based on chaining distance between graphs of neighbors, as described in [1].

Parameters
----------
$K_PARAM

$KNN_PARAMS

Examples
--------
$(SCORE_UNSUPERVISED("COFDetector"))

References
----------
[1] Tang, Jian; Chen, Zhixiang; Fu, Ada Wai-Chee; Cheung, David Wai-Lok (2002): Enhancing Effectiveness of Outlier
Detections for Low Density Patterns.
"""
OD.@detector mutable struct COFDetector <: UnsupervisedDetector
    k::Integer = 5::(_ > 0)
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree, :brutetree))
    static::Union{Bool,Symbol} = :auto::(_ in (true, false, :auto))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
end

struct COFModel <: DetectorModel
    # An efficient COF prediction requires us to store the full pairwise distance matrix of the training examples in
    # addition to the learned tree as well as the ACDs of the training examples.
    tree::NN.NNTree
    pdists::AbstractArray
    acds::AbstractVector
end

function OD.fit(detector::COFDetector, X::Data; verbosity)::Fit
    # calculate pairwise distances in addition to building the tree;
    # TODO: we could remove this once NearestNeighbors.jl exports something like `allpairs`
    pdists = DI.pairwise(detector.metric, X, dims = 2)

    # Note: Fitting is different from pyOD, because we ignore the trivial nearest neighbor using knn_others as in
    # all other nearest-neighbor-based algorithms
    X = prepare_data(X, detector.static)

    # use tree to calculate distances
    tree = @tree detector X

    # we need k + 1 neighbors to calculate the chaining distance and have to make sure the indices are sorted 
    idxs, _ = detector.parallel ?
              knn_parallel(tree, X, detector.k + 1, true) :
              knn_sequential(tree, X, detector.k + 1, true)

    acds = _acd(idxs, pdists, detector.k)
    scores = _cof(idxs, acds, detector.k)
    return COFModel(tree, pdists, acds), scores
end

function OD.transform(detector::COFDetector, model::COFModel, X::Data)::Scores
    X = prepare_data(X, detector.static)

    # Note: It's important to sort the neighbors, because _calc_acds depends on the order of the neighbors
    idxs, _ = detector.parallel ?
              knn_parallel(model.tree, X, detector.k + 1, false, true) :
              knn_sequential(model.tree, X, detector.k + 1, false, true)

    return _cof(idxs, model.pdists, model.acds, detector.k)
end

function _cof(idxs::AbstractVector{<:AbstractVector}, acds::AbstractVector, k::Int)::Scores
    # Calculate the connectivity-based outlier factor from given acds
    cof = Vector{Float64}(undef, length(idxs))
    for (i, idx) in enumerate(idxs)
        @inbounds cof[i] = (acds[i] * k) / sum(acds[idx[2:end]])
    end
    cof
end

function _cof(idxs::AbstractVector{<:AbstractVector}, pdists::AbstractMatrix, acds::AbstractVector, k::Int)::Scores
    # Calculate the connectivity-based outlier factor for test examples with given training distances and acds.
    cof = Vector{Float64}(undef, length(idxs))
    acdsTest = _acd(idxs, pdists, k)
    for (i, idx) in enumerate(idxs)
        @inbounds cof[i] = (acdsTest[i] * k) / sum(acdsTest[idx[2:end]])
    end
    cof
end

function _acd(idxs::AbstractVector{<:AbstractVector}, pdists::AbstractMatrix, k::Int)::Vector{Float64}
    # Initialize with zeros because we add to each entry
    acds = zeros(Float64, length(idxs))
    kplus1 = k + 1
    for (i, idx) in enumerate(idxs)
        for j in 1:k
            # calculate the minimum distance (from all reachable points). That is, we sort the distances of a specific
            # point (given by idx[j+1]) according to the order of the current idx, where idx[1] specifies the idx of the
            # nearest neighbors and idx[k] specifies the idx of the k-th neighbor. We then restrict this so-called
            # set-based nearest path (SBN) to the points that are reachable with [begin:j]
            cost = minimum(pdists[idx, idx[j+1]][begin:j])
            @inbounds acds[i] += ((2.0 * (kplus1 - j)) / (k * kplus1)) * cost
        end
    end
    acds
end
