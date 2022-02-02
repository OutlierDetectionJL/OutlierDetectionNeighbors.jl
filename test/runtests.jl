using OutlierDetectionNeighbors
using OutlierDetectionTest
using Distances: Cityblock

# Test the metadata of all exported detectors
test_meta.(eval.(OutlierDetectionNeighbors.MODELS))

data = TestData()
run_test(detector) = test_detector(detector, data)

# ABOD
run_test(ABODDetector())
run_test(ABODDetector(static=false))
run_test(ABODDetector(parallel=true, enhanced=true))

# COF
run_test(COFDetector())
run_test(COFDetector(metric=Cityblock()))
run_test(COFDetector(static=false))
run_test(COFDetector(parallel=true))

# DNN
run_test(DNNDetector(d=1))
run_test(DNNDetector(d=1, static=false))
run_test(DNNDetector(d=1, parallel=true))

# KNN
run_test(KNNDetector(reduction=:maximum))
run_test(KNNDetector(static=false))
run_test(KNNDetector(parallel=true, reduction=:mean, algorithm=:balltree))
run_test(KNNDetector(parallel=true, reduction=:median))

# LOF
run_test(LOFDetector())
run_test(LOFDetector(static=false))
run_test(LOFDetector(parallel=true))
run_test(LOFDetector(static=false, parallel=true, algorithm=:brutetree))
