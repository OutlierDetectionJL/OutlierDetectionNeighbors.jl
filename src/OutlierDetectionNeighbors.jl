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

    MODELS = (ABODDetector,
              COFDetector,
              DNNDetector,
              KNNDetector,
              LOFDetector)

    org = "OutlierDetectionJL"
    uuid = "51249a0a-cb36-4849-8e04-30c7f8d311bb"
    for model in MODELS
        @eval(export $model)
        OD.metadata_pkg(model, package_name=@__MODULE__, package_uuid=uuid,
                        package_url="https://github.com/$org/$(@__MODULE__).jl",
                        is_pure_julia=true, package_license="MIT", is_wrapper=false)
        OD.load_path(::Type{model}) = "$(@__MODULE__).$model"
    end
end
