module OutlierDetectionNeighbors
    using OutlierDetectionInterface
    using OutlierDetectionInterface:SCORE_UNSUPERVISED
    const OD = OutlierDetectionInterface

    import NearestNeighbors
    const NN = NearestNeighbors

    import Distances
    const DI = Distances

    include("utils.jl")
    include("models/abod.jl")
    include("models/cof.jl")
    include("models/dnn.jl")
    include("models/knn.jl")
    include("models/lof.jl")

    const UUID = "51249a0a-cb36-4849-8e04-30c7f8d311bb"
    const MODELS = [:ABODDetector,
                    :COFDetector,
                    :DNNDetector,
                    :KNNDetector,
                    :LOFDetector]

    for model in MODELS
        @eval begin
            OD.@default_frontend $model
            OD.@default_metadata $model $UUID
            export $model
        end
    end

"""
`ABODDetector`: Determine outliers based on the angles to its nearest neighbors. This implements the FastABOD variant described in the paper, that is, it uses the variance of angles to its nearest neighbors, not to the whole dataset.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X, y)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `k::Integer=5`: The number of neighbors to use. Must be greater than zero.
- `metric::Distances.Metric=Distances.Euclidean`: The distance metric to use,
  as defined in the `Distances.jl` package
- `algorithm::Symbol=:kdtree` The algorithm to use. One of (:kdtree,
  :balltree). In a kdtree, points are recursively split into groups using
  hyper-planes. Therefore a KDTree only works with axis aligned metrics which
  are: Euclidean, Chebyshev, Minkowski and Cityblock. A brutetree linearly
  searches all points in a brute force fashion and works with any Metric. A
  balltree recursively splits points into groups bounded by hyper-spheres and
  works with any Metric.
- `static::Union{Bool, Symbol}=:auto`: Whether the input data should be
  statically or dynamically allocated. Can either be `true`, `false`, or
  `:auto`. If true, the data is statically allocated. If false, the data is
  dynamically allocated. If :auto, the data is dynamically allocated if the
  product of all dimensions except the last is greater than 100.
- `leafsize::Integer=10`: Determines at what number of points to stop
  splitting the tree further. There is a trade-off between traversing the
  tree and having to evaluate the metric function for increasing number of
  points.
- `reorder::Bool=true`While building the tree this will put points close in
  distance close in memory since this helps with cache locality. In this
  case, a copy of the original data will be made so that the original data is
  left unmodified. This can have a significant impact on performance and is
  by default set to true.
- `parallel::Bool=false`: Parallelize score and predict using all threads
  available. The number of threads can be set with the JULIA_NUM_THREADS
  environment variable. Note: fit is not parallel.

# Operations

- `transform(mach, Xnew)`: Return a transformed matrix of type
  `ScientificTypesBase.Continuous` given new features `Xnew`.

# Examples

```julia
using MLJ, OutlierDetection
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

ABODDetector = @load ABODDetector pkg=OutlierDetectionNeighbors
detector = ABODDetector()
mach = machine(detector, X)
test_scores = transform(detector, result.model, X)
```

"""
ABODDetector

"""
`COFDetector`: Determine local outliers using density based on chaining distance between graphs of neighbors.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X, y)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `k::Integer=5`: The number of neighbors to use. Must be greater than zero.
- `metric::Distances.Metric=Distances.Euclidean`: The distance metric to use,
  as defined in the `Distances.jl` package
- `algorithm::Symbol=:kdtree` The algorithm to use. One of (:kdtree,
  :balltree). In a kdtree, points are recursively split into groups using
  hyper-planes. Therefore a KDTree only works with axis aligned metrics which
  are: Euclidean, Chebyshev, Minkowski and Cityblock. A brutetree linearly
  searches all points in a brute force fashion and works with any Metric. A
  balltree recursively splits points into groups bounded by hyper-spheres and
  works with any Metric.
- `static::Union{Bool, Symbol}=:auto`: Whether the input data should be
  statically or dynamically allocated. Can either be `true`, `false`, or
  `:auto`. If true, the data is statically allocated. If false, the data is
  dynamically allocated. If :auto, the data is dynamically allocated if the
  product of all dimensions except the last is greater than 100.
- `leafsize::Integer=10`: Determines at what number of points to stop
  splitting the tree further. There is a trade-off between traversing the
  tree and having to evaluate the metric function for increasing number of
  points.
- `reorder::Bool=true`While building the tree this will put points close in
  distance close in memory since this helps with cache locality. In this
  case, a copy of the original data will be made so that the original data is
  left unmodified. This can have a significant impact on performance and is
  by default set to true.
- `parallel::Bool=false`: Parallelize score and predict using all threads
  available. The number of threads can be set with the JULIA_NUM_THREADS
  environment variable. Note: fit is not parallel.

# Operations

- `transform(mach, Xnew)`: Return a transformed matrix of type
  `ScientificTypesBase.Continuous` given new features `Xnew`.

# Examples

```julia
using MLJ, OutlierDetection
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

COFDetector = @load COFDetector pkg=OutlierDetectionNeighbors
detector = COFDetector()
mach = machine(detector, X)
test_scores = transform(detector, result.model, X)
```
"""
COFDetector

"""
`DNNDetector`: Identify anomalies based on the number of neighbors in a hypersphere of radius `d`.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X, y)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `d::Real=0`:The hypersphere radius used to calculate the global density of an instance. Must be greater than zero, will give a warning if unset
- `metric::Distances.Metric=Distances.Euclidean`: The distance metric to use,
  as defined in the `Distances.jl` package
- `algorithm::Symbol=:kdtree` The algorithm to use. One of (:kdtree,
  :balltree). In a kdtree, points are recursively split into groups using
  hyper-planes. Therefore a KDTree only works with axis aligned metrics which
  are: Euclidean, Chebyshev, Minkowski and Cityblock. A brutetree linearly
  searches all points in a brute force fashion and works with any Metric. A
  balltree recursively splits points into groups bounded by hyper-spheres and
  works with any Metric.
- `static::Union{Bool, Symbol}=:auto`: Whether the input data should be
  statically or dynamically allocated. Can either be `true`, `false`, or
  `:auto`. If true, the data is statically allocated. If false, the data is
  dynamically allocated. If :auto, the data is dynamically allocated if the
  product of all dimensions except the last is greater than 100.
- `leafsize::Integer=10`: Determines at what number of points to stop
  splitting the tree further. There is a trade-off between traversing the
  tree and having to evaluate the metric function for increasing number of
  points.
- `reorder::Bool=true`While building the tree this will put points close in
  distance close in memory since this helps with cache locality. In this
  case, a copy of the original data will be made so that the original data is
  left unmodified. This can have a significant impact on performance and is
  by default set to true.
- `parallel::Bool=false`: Parallelize score and predict using all threads
  available. The number of threads can be set with the JULIA_NUM_THREADS
  environment variable. Note: fit is not parallel.

# Operations

- `transform(mach, Xnew)`: Return a transformed matrix of type
  `ScientificTypesBase.Continuous` given new features `Xnew`.

# Examples

```julia
using MLJ, OutlierDetection
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

DNNDetector = @load DNNDetector pkg=OutlierDetectionNeighbors
detector = DNNDetector(d=0.5)
mach = machine(detector, X)
test_scores = transform(detector, result.model, X)
```
"""
DNNDetector
"""
`KNNDetector`: Identify anomalies using the distances between an instance and its k nearest neighbors.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X, y)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `k::Integer=5`: The number of neighbors to use. Must be greater than zero.
- `metric::Distances.Metric=Distances.Euclidean`: The distance metric to use,
  as defined in the `Distances.jl` package
- `algorithm::Symbol=:kdtree` The algorithm to use. One of (:kdtree,
  :balltree). In a kdtree, points are recursively split into groups using
  hyper-planes. Therefore a KDTree only works with axis aligned metrics which
  are: Euclidean, Chebyshev, Minkowski and Cityblock. A brutetree linearly
  searches all points in a brute force fashion and works with any Metric. A
  balltree recursively splits points into groups bounded by hyper-spheres and
  works with any Metric.
- `static::Union{Bool, Symbol}=:auto`: Whether the input data should be
  statically or dynamically allocated. Can either be `true`, `false`, or
  `:auto`. If true, the data is statically allocated. If false, the data is
  dynamically allocated. If :auto, the data is dynamically allocated if the
  product of all dimensions except the last is greater than 100.
- `leafsize::Integer=10`: Determines at what number of points to stop
  splitting the tree further. There is a trade-off between traversing the
  tree and having to evaluate the metric function for increasing number of
  points.
- `reorder::Bool=true`While building the tree this will put points close in
  distance close in memory since this helps with cache locality. In this
  case, a copy of the original data will be made so that the original data is
  left unmodified. This can have a significant impact on performance and is
  by default set to true.
- `parallel::Bool=false`: Parallelize score and predict using all threads
  available. The number of threads can be set with the JULIA_NUM_THREADS
  environment variable. Note: fit is not parallel.
- `reduction::Symbol=:maximum` The reduction method to use on the neighborhood distances. One of (`:maximum`, `:mean`, `:median`).


# Operations

- `transform(mach, Xnew)`: Return a transformed matrix of type
  `ScientificTypesBase.Continuous` given new features `Xnew`.

# Examples

```julia
using MLJ, OutlierDetection
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

KNNDetector = @load KNNDetector pkg=OutlierDetectionNeighbors
detector = KNNDetector(reduction=:mean)
mach = machine(detector, X)
test_scores = transform(detector, result.model, X)
```
"""
KNNDetector

"""
`LOFDetector`: Identify anomalies using the density of an instance in comparison to its neighbors. This algorithm introduced the notion of local outliers.

# Training data

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X, y)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


# Hyper-parameters

- `k::Integer=5`: The number of neighbors to use. Must be greater than zero.
- `metric::Distances.Metric=Distances.Euclidean`: The distance metric to use,
  as defined in the `Distances.jl` package
- `algorithm::Symbol=:kdtree` The algorithm to use. One of (:kdtree,
  :balltree). In a kdtree, points are recursively split into groups using
  hyper-planes. Therefore a KDTree only works with axis aligned metrics which
  are: Euclidean, Chebyshev, Minkowski and Cityblock. A brutetree linearly
  searches all points in a brute force fashion and works with any Metric. A
  balltree recursively splits points into groups bounded by hyper-spheres and
  works with any Metric.
- `static::Union{Bool, Symbol}=:auto`: Whether the input data should be
  statically or dynamically allocated. Can either be `true`, `false`, or
  `:auto`. If true, the data is statically allocated. If false, the data is
  dynamically allocated. If :auto, the data is dynamically allocated if the
  product of all dimensions except the last is greater than 100.
- `leafsize::Integer=10`: Determines at what number of points to stop
  splitting the tree further. There is a trade-off between traversing the
  tree and having to evaluate the metric function for increasing number of
  points.
- `reorder::Bool=true`While building the tree this will put points close in
  distance close in memory since this helps with cache locality. In this
  case, a copy of the original data will be made so that the original data is
  left unmodified. This can have a significant impact on performance and is
  by default set to true.
- `parallel::Bool=false`: Parallelize score and predict using all threads
  available. The number of threads can be set with the JULIA_NUM_THREADS
  environment variable. Note: fit is not parallel.

# Operations

- `transform(mach, Xnew)`: Return a transformed matrix of type
  `ScientificTypesBase.Continuous` given new features `Xnew`.

# Examples

```julia
using MLJ, OutlierDetection
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

LOFDetector = @load LOFDetector pkg=OutlierDetectionNeighbors
detector = LOFDetector()
mach = machine(detector, X)
test_scores = transform(detector, result.model, X)
```
"""
LOFDetector

end
