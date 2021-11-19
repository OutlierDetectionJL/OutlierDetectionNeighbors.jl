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
        $detector.algorithm === :kdtree ?
            NN.KDTree($X, $detector.metric; $detector.leafsize, $detector.reorder) :
            NN.BallTree($X, $detector.metric; $detector.leafsize, $detector.reorder)
    end)
end

function knn_parallel(tree::NN.NNTree, X::AbstractArray, k::Int,
    sort::Bool = false)::Tuple{AbstractVector, AbstractVector}
    # pre-allocate the result arrays (as in NearestNeighbors.jl)
    samples = size(X, 2)
    dists = [Vector{NN.get_T(eltype(X))}(undef, k) for _ in 1:samples]
    idxs = [Vector{Int}(undef, k) for _ in 1:samples]

    # get number of threads
    nThreads = Threads.nthreads()
    # partition the input array equally
    partition_size = samples ÷ nThreads + 1
    partitions = Iterators.partition(axes(X, 2), partition_size)
    Threads.@threads for idx = collect(partitions)
        @inbounds idxs[idx], dists[idx] = NN.knn(tree, view(X, :, idx), k, sort)
    end
    idxs, dists
end

function dnn_parallel(tree::NN.NNTree, X::AbstractArray, d::Real, sort::Bool = false)::AbstractVector
    # pre-allocate the result arrays (as in NearestNeighbors.jl)
    samples = size(X, 2)
    scores = Vector{Float64}(undef, samples)

    # get number of threads
    nThreads = Threads.nthreads()
    # partition the input array equally
    partition_size = samples ÷ nThreads + 1
    partitions = Iterators.partition(axes(X, 2), partition_size)
    Threads.@threads for idx = collect(partitions)
        @inbounds scores[idx] = dnn(NN.inrange(tree, view(X, :, idx), d, sort))
    end
    scores
end

# Calculate the k-nearest neighbors with ignoring the own point in the tree.
function knn_others(tree::NN.NNTree, X::AbstractArray, k::Integer)::Tuple{AbstractVector, AbstractVector}
    idxs, dists = NN.knn(tree, X, k + 1, true) # we ignore the distance to the 'self' point, important to sort!
    ignore_self = vecvec -> map(vec -> vec[2:end], vecvec)
    ignore_self(idxs), ignore_self(dists)
end

# The NN package automatically converts matrices to a vector of points (static vectors) for improved performance
# this results in very bad performance for high-dimensional matrices (e.g. d > 100). 
dynamic_view(X::Data) = [NN.SizedVector{length(v)}(v) for v in eachslice(X; dims=ndims(X))]
auto_view(X::Data) = prod(size(X)[1:end-1]) > 100 ? dynamic_view(X) : X
prepare_data(X::Data, static::Union{Bool, Symbol}) = static === :auto ? auto_view(X) :
                                                     static === false ? dynamic_view(X) : X
