using OutlierDetectionNeighbors
using OutlierDetectionTest

# Test the metadata of all exported detectors
test_meta.(eval.(OutlierDetectionNeighbors.MODELS))

data = TestData()
run_test(detector) = test_detector(detector, data)

# ABOD
run_test(ABODDetector())
run_test(ABODDetector(parallel=true, enhanced=true))

# COF
run_test(COFDetector())
run_test(COFDetector(parallel=true))

# DNN
run_test(DNNDetector(d=1))
run_test(DNNDetector(d=1, parallel=true))

# KNN
run_test(KNNDetector(reduction=:maximum))
run_test(KNNDetector(parallel=true, reduction=:mean, algorithm=:balltree))
run_test(KNNDetector(parallel=true, reduction=:median))

# LOF
run_test(LOFDetector())
run_test(LOFDetector(parallel=true))
