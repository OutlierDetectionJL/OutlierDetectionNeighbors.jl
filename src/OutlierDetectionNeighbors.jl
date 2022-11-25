module OutlierDetectionNeighbors
    using OutlierDetectionInterface
    using OutlierDetectionInterface:SCORE_UNSUPERVISED
    const OD = OutlierDetectionInterface

    import NearestNeighbors
    import MLJModelInterface
    const NN = NearestNeighbors

    import Distances
    const DI = Distances

    include("utils.jl")
    include("models/abod.jl")
    include("models/cof.jl")
    include("models/dnn.jl")
    include("models/knn.jl")
    include("models/lof.jl")
    include("models/docstrings.jl")

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
end
