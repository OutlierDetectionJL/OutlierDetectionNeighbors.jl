"""
    DNNDetector(d = 0,
                metric = Euclidean(),
                algorithm = :kdtree,
                leafsize = 10,
                reorder = true,
                parallel = false)

Anomaly score based on the number of neighbors in a hypersphere of radius `d`. Knorr et al. [1] directly converted the
resulting outlier scores to labels, thus this implementation does not fully reflect the approach from the paper.

Parameters
----------
    d::Real
The hypersphere radius used to calculate the global density of an instance.

$KNN_PARAMS

Examples
--------
$(SCORE_UNSUPERVISED("DNNDetector"))

References
----------
[1] Knorr, Edwin M.; Ng, Raymond T. (1998): Algorithms for Mining Distance-Based Outliers in Large Datasets.
"""
OD.@detector mutable struct DNNDetector <: UnsupervisedDetector
    metric::DI.Metric = DI.Euclidean()
    algorithm::Symbol = :kdtree::(_ in (:kdtree, :balltree, :brutetree))
    static::Union{Bool,Symbol} = :auto::(_ in (true, false, :auto))
    leafsize::Integer = 10::(_ ≥ 0)
    reorder::Bool = true
    parallel::Bool = false
    d::Real = 0::(_ > 0) # warns if `d` is not set
end

struct DNNModel <: DetectorModel
    tree::NN.NNTree
end

function OD.fit(detector::DNNDetector, X::Data; verbosity)::Fit
    X = prepare_data(X, detector.static)

    # create the specified tree
    tree = @tree detector X

    # use tree to calculate distances
    scores = detector.parallel ?
             dnn_parallel(tree, X, detector.d, true) :
             dnn_sequential(tree, X, detector.d, true)
    return DNNModel(tree), scores
end

function OD.transform(detector::DNNDetector, model::DNNModel, X::Data)::Scores
    X = prepare_data(X, detector.static)
    return detector.parallel ?
           dnn_parallel(model.tree, X, detector.d) :
           dnn_sequential(model.tree, X, detector.d)
end

@inline function _dnn(idxs::AbstractVector{<:AbstractVector})::Scores
    # Helper function to reduce the instances to a global density score.
    1 ./ (length.(idxs) .+ 0.1) # min score = 0, max_score = 10
end

@inline function _dnn_others(idxs::AbstractVector{<:AbstractVector})::Scores
    # Remove the (self) point previously added when fitting the tree, otherwise during `fit`, that point would always
    # be included in the density estimation
    1 ./ (length.(idxs) .- 0.9) # 1 - 0.1
end

function dnn_sequential(tree::NN.NNTree, X::AbstractVector{<:AbstractVector}, d::Real, ignore_self::Bool = false)
    # returns a vector of vectors containing the nearest indices
    idxs = NN.inrange(tree, X, d, false)
    ignore_self ? _dnn_others(idxs) : _dnn(idxs)
end

function dnn_parallel(tree::NN.NNTree, X::AbstractVector{<:AbstractVector}, d::Real, ignore_self::Bool = false)
    # returns a vector of vectors containing the nearest indices
    samples = length(X)
    scores = Vector{Float64}(undef, samples)

    # get number of threads
    nThreads = Threads.nthreads()
    # partition the input array equally
    partition_size = samples ÷ nThreads + 1
    partitions = Iterators.partition(eachindex(X), partition_size)
    dnn_closure(idxs) = ignore_self ? _dnn_others(idxs) : _dnn(idxs)

    Threads.@threads for idx = collect(partitions)
        @inbounds scores[idx] = dnn_closure(NN.inrange(tree, X[idx], d, false))
    end
    return scores
end
