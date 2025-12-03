const K_PARAM = """    k::Integer
Number of neighbors (must be greater than 0)."""

const KNN_PARAMS = """    metric::Metric
This is one of the Metric types defined in the Distances.jl package. It is possible to define your own metrics by
creating new types that are subtypes of Metric.

    algorithm::Symbol
One of `(:kdtree, :balltree)`. In a `kdtree`, points are recursively split into groups using hyper-planes.
Therefore a KDTree only works with axis aligned metrics which are: Euclidean, Chebyshev, Minkowski and Cityblock.
A *brutetree* linearly searches all points in a brute force fashion and works with any Metric. A *balltree*
recursively splits points into groups bounded by hyper-spheres and works with any Metric.

    static::Union{Bool, Symbol}
One of `(true, false, :auto)`. Whether the input data for fitting and transform should be statically or dynamically
allocated. If `true`, the data is statically allocated. If `false`, the data is dynamically allocated. If `:auto`,
the data is dynamically allocated if the product of all dimensions except the last is greater than 100.

    leafsize::Int
Determines at what number of points to stop splitting the tree further. There is a trade-off between traversing the
tree and having to evaluate the metric function for increasing number of points.

    reorder::Bool
While building the tree this will put points close in distance close in memory since this helps with cache locality.
In this case, a copy of the original data will be made so that the original data is left unmodified. This can have a
significant impact on performance and is by default set to true.

    parallel::Bool
Parallelize `score` and `predict` using all threads available. The number of threads can be set with the
`JULIA_NUM_THREADS` environment variable. Note: `fit` is not parallel."""

# for some reason providing this as a function is massively slowing down compilation
macro tree(detector, X)
    esc(quote
        if $detector.algorithm === :kdtree
            NN.KDTree($X, $detector.metric; $detector.leafsize, $detector.reorder)
        elseif $detector.algorithm === :balltree
            NN.BallTree($X, $detector.metric; $detector.leafsize, $detector.reorder)
        elseif $detector.algorithm === :brutetree
            # intentionally not propagating leafsize and reorder, because that may lead to problems
            NN.BruteTree($X, $detector.metric)
        end
    end)
end

function knn_parallel(tree::NN.NNTree, X::AbstractVector{<:AbstractVector},
    k::Int, ignore_self = false, sort::Bool = false)
    # pre-allocate the result arrays (as in NearestNeighbors.jl)
    indices = eachindex(X)
    dists = [Vector{NN.get_T(eltype(X))}(undef, k) for _ in indices]
    idxs = [Vector{Int}(undef, k) for _ in indices]

    # get number of threads
    nThreads = Threads.nthreads()

    # partition the input array equally
    n_samples = length(X)
    divides_data = mod(n_samples, nThreads) == 0
    partition_size = divides_data ? n_samples ÷ nThreads : n_samples ÷ nThreads + 1
    partitions = Iterators.partition(indices, partition_size)

    # create the knn function depending on whether we need to ignore self
    knn_closure(tree, X, k, sort) = ignore_self ? _knn_others(tree, X, k) : NN.knn(tree, X, k, sort)

    # parallel computation over the equal array splits
    Threads.@threads for idx = collect(partitions)
        @inbounds idxs[idx], dists[idx] = knn_closure(tree, X[idx], k, sort)
    end
    idxs, dists
end

function knn_sequential(tree::NN.NNTree, X::AbstractVector{<:AbstractVector},
    k::Int, ignore_self = false, sort::Bool = false)
    ignore_self ? _knn_others(tree, X, k) : NN.knn(tree, X, k, sort)
end

# Calculate the k-nearest neighbors with ignoring the own point in the tree.
function _knn_others(tree::NN.NNTree, X::AbstractVector, k::Integer)
    idxs, dists = NN.knn(tree, X, k + 1, true) # we ignore the distance to the 'self' point, important to sort!
    ignore_self = vecvec -> map(vec -> vec[2:end], vecvec)
    ignore_self(idxs), ignore_self(dists)
end

# The NN package automatically converts matrices to a vector of points (static vectors) for improved performance
# this results in very bad performance for high-dimensional matrices (e.g. d > 100).
dynamic_view(X::Data) = [SA.SizedVector{length(v)}(v) for v in eachslice(X; dims = ndims(X))]
static_view(X::Data) = [SA.SVector{length(v)}(v) for v in eachslice(X; dims = ndims(X))]
auto_view(X::Data) = prod(size(X)[1:end-1]) > 100 ? dynamic_view(X) : static_view(X)
function prepare_data(X::Data, static::Union{Bool,Symbol})
    @assert ndims(X) == 2 "k-NN currently only supports matrices."
    return static === :auto ? auto_view(X) :
           static ? static_view(X) : dynamic_view(X)
end
