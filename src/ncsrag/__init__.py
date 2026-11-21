"""ncsrag: noise-conditional selective generation for retrieval-augmented LMs.

Companion library to "Knowing When Not to Answer" (ACML 2026, journal track).
"""
from .conformal import (RiskBinnedCRC, BinCertificate, clopper_pearson_upper,
                        decertify_on_drift, marginal_threshold)
from .estimator import (ALL_FEATURES, CONTRAST_FEATURES, MODEL_FEATURES, RETRIEVAL_FEATURES, answered_rows,
                        hallucination_labels, out_of_fold_risk)
from .protocol import NOISY_CONDITIONS, run_noise_shift

__version__ = "1.0.0"
__all__ = ["RiskBinnedCRC", "BinCertificate", "clopper_pearson_upper", "decertify_on_drift",
           "marginal_threshold", "ALL_FEATURES", "CONTRAST_FEATURES", "MODEL_FEATURES", "RETRIEVAL_FEATURES",
           "answered_rows", "hallucination_labels", "out_of_fold_risk",
           "NOISY_CONDITIONS", "run_noise_shift", "__version__"]
