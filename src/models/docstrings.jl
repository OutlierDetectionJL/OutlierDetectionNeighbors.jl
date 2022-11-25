# # HELPERS

# header for mlj part of document strings
mlj_header(T) =
    """
    # MLJ Interface

    $MLJModelInterface.doc_header(T, augment=true))

    """


# # ABODDetector

"""
    ABODDetector(k = 5,
                 metric = Euclidean(),
                 algorithm = :kdtree,
                 static = :auto,
                 leafsize = 10,
                 reorder = true,
                 parallel = false,
                 enhanced = false)

Determine outliers based on the angles to its nearest neighbors. This implements the `FastABOD` variant described in
the paper, that is, it uses the variance of angles to its nearest neighbors, not to the whole dataset, see [1]. 

*Notice:* The scores are inverted, to conform to our notion that higher scores describe higher outlierness.

Parameters
----------
$K_PARAM

$KNN_PARAMS

    enhanced::Bool
When `enhanced=true`, it uses the enhanced ABOD (EABOD) adaptation proposed by [2].

Examples (native interface)
---------------------------
$(SCORE_UNSUPERVISED("ABODDetector"))

References
----------
[1] Kriegel, Hans-Peter; S hubert, Matthias; Zimek, Arthur (2008): Angle-based outlier detection in high-dimensional
data.

[2] Li, Xiaojie; Lv, Jian Cheng; Cheng, Dongdong (2015): Angle-Based Outlier Detection Algorithm with More Stable
Relationships.

$(mlj_header(ABODDetector))

In MLJ or MLJBase, bind an instance `model` to data with:

```
    mach = machine(model, X)
```

where

- `X`: any table of input features (eg, a `DataFrame`) whose columns each have
  `Continuous` element scitype; check column scitypes with `schema(X)`

Train the machine using `fit!(mach, rows=...)`.


## Operations

- `transform(mach, Xnew)`: Return a transformed vector of scores (element scitype
  `Continuous`); here `Xnew` should have the same scitype as `X`.

## Examples

```julia
using MLJ
import OutlierDetectionData
X, y = OutlierDetectionData.ODDS.load("annthyroid")

ABODDetector = @load ABODDetector pkg=OutlierDetectionNeighbors
detector = ABODDetector()
mach = machine(detector, X) |> fit!
test_scores = transform(mach, X)
```
"""
ABODDetector
