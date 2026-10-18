"""CSP + LDA baseline (MNE ``CSP`` log-variance features, scikit-learn ``LDA``).

Deterministic, no seeds. ``fit_predict`` returns ``(n_test, 2)`` pseudo-logits whose
softmax equals the LDA posterior, so ``src.eval.compute_kpis`` applies unchanged.
"""

from __future__ import annotations

import mne
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def fit_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_components: int = 6) -> np.ndarray:
    mne.set_log_level("WARNING")
    csp = CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis()
    Xtr = np.asarray(X_train, dtype=np.float64)
    Xte = np.asarray(X_test, dtype=np.float64)
    feats = csp.fit_transform(Xtr, np.asarray(y_train))
    lda.fit(feats, y_train)
    proba = lda.predict_proba(csp.transform(Xte))
    return np.log(np.clip(proba, 1e-12, 1.0)).astype(np.float32)
