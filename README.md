# pivot-selection
__*select_pivots*__: pivot selection methods for sentiment analysis
- Methods: FREQ, MI, PMI
- Datasets: L, U

__*landmark_pivot*__: landmark-based pivot selection method
- Pre-defined: PPMI
- Datasets: S_L, T_U (UDA with SPS)


similarity measurement
----------------
__*compare_ranking*__: similarity of selected pivots **between datasets** and **among methods**
- Intersection: Jaccard Coeffients
- Ordering: Kandell Rank Coeffients


performance evaluation
-----------------
__*evaluate_scl/sfa*__: performance on DA methods by applied pivot selection methods
- Testing Measure: Classification Accuracy, Clopper Pearson
- DA methods: SCL, SFA
