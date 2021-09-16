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
            OD.metadata_pkg($model, package_name=string(@__MODULE__), package_uuid=$UUID,
                            package_url="https://github.com/OutlierDetectionJL/$(@__MODULE__).jl",
                            is_pure_julia=true, package_license="MIT", is_wrapper=false)
            OD.load_path(::Type{$model}) = string($model)
            export $model
        end
    end
end
