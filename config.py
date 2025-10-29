"""
config.py

Shared configuration parameters for latent polytree experiments.
"""

# Random seeds
DEFAULT_SEED = 42

# Numerical tolerance
NUMERICAL_EPS = 1e-7

# Default trial counts
DEFAULT_N_TRIALS = 20
DEFAULT_N_TRIALS_POPULATION = 30

# Sample size ranges for finite-sample analysis
DEFAULT_SAMPLE_SIZES = [100, 1000, 10000, 100000, 500000, 1000000, 10000000]

# Polytree generation parameters
DEFAULT_WEIGHTS_RANGE = (0.8, 1.2)
DEFAULT_N_LATENT = 1

# Edge weight thresholds for population experiments
EDGE_WEIGHT_THRESHOLDS = [0.1, 0.3, 0.5, 0.8]

# Polytree sizes for scaling analysis
SMALL_POLYTREE_SIZES = [4, 5, 6, 7, 8]
MEDIUM_POLYTREE_SIZES = [10, 15, 20, 25, 30]
LARGE_POLYTREE_SIZES = [40, 50, 60, 70, 80, 90, 100]

# Performance thresholds
F1_EXCELLENT_THRESHOLD = 0.9
F1_GOOD_THRESHOLD = 0.7
F1_MODERATE_THRESHOLD = 0.5

# Noise distribution parameters
GAMMA_DISTRIBUTION = "gamma"
UNIFORM_DISTRIBUTION = "uniform"
DEFAULT_NOISE_DISTRIBUTION = GAMMA_DISTRIBUTION

# Output directories
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
DATA_DIR = "data"

# File naming conventions
POPULATION_RESULTS_PREFIX = "population_analysis"
FINITE_SAMPLE_RESULTS_PREFIX = "finite_sample_analysis"
BREAKDOWN_RESULTS_PREFIX = "breakdown_analysis"

# Plotting parameters
FIGURE_DPI = 300
FIGURE_FORMAT_PDF = "pdf"
FIGURE_FORMAT_PNG = "png"