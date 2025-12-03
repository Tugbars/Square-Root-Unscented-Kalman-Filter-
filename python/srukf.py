"""
Student-t Square-Root UKF and Kelly Criterion - Python Bindings

Usage:
    from srukf import StudentTSRUKF, Kelly
    
    # Create filter
    ukf = StudentTSRUKF(nx=3, nz=1, nu=4.0)
    ukf.set_state([100.0, 0.0, -3.0])
    ukf.set_dynamics(np.eye(3))
    ukf.set_measurement(np.array([[1, 0, 0]]))
    
    # Run filter
    for z in measurements:
        ukf.step(z)
        print(ukf.state, ukf.nis)
    
    # Kelly sizing
    kelly = Kelly.from_ukf(ukf, fraction=0.5)
    print(kelly.position, kelly.sharpe)
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple, Union

# =============================================================================
# Windows MKL DLL paths (must be set BEFORE loading the library)
# =============================================================================

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

# =============================================================================
# Load shared library
# =============================================================================

def _load_library():
    """Find and load the SRUKF shared library."""
    search_paths = [
        Path(__file__).parent / "student_t_srukf.dll",
        Path(__file__).parent / "libstudent_t_srukf.dll",
        Path(__file__).parent / "student_t_srukf.so",
        Path(__file__).parent / "libstudent_t_srukf.so",
        Path(".") / "student_t_srukf.dll",
        Path(".") / "libstudent_t_srukf.dll",
        Path("./build/Release") / "student_t_srukf.dll",
        Path("./build/Debug") / "student_t_srukf.dll",
        "student_t_srukf.dll",
        "libstudent_t_srukf.so",
    ]
    
    for path in search_paths:
        try:
            return ctypes.CDLL(str(path))
        except OSError:
            continue
    
    raise RuntimeError(
        "Could not load student_t_srukf.dll/.so. Build it with:\n"
        "  cmake --build build --target student_t_srukf_shared --config Release\n"
        "Then copy to this directory or add to PATH."
    )

_lib = _load_library()

# =============================================================================
# C Function Signatures
# =============================================================================

# Pointer types
c_double_p = ctypes.POINTER(ctypes.c_double)
c_int_p = ctypes.POINTER(ctypes.c_int)
c_void_p = ctypes.c_void_p

def _setup_signatures():
    """Set up ctypes function signatures for type safety."""
    
    # srukf_create(int nx, int nz, double nu) -> StudentT_SRUKF*
    _lib.srukf_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double]
    _lib.srukf_create.restype = c_void_p
    
    # srukf_destroy(StudentT_SRUKF* ukf)
    _lib.srukf_destroy.argtypes = [c_void_p]
    _lib.srukf_destroy.restype = None
    
    # srukf_set_state(ukf, x)
    _lib.srukf_set_state.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_state.restype = None
    
    # srukf_set_sqrt_cov(ukf, S)
    _lib.srukf_set_sqrt_cov.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_sqrt_cov.restype = None
    
    # srukf_set_dynamics(ukf, F)
    _lib.srukf_set_dynamics.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_dynamics.restype = None
    
    # srukf_set_measurement(ukf, H)
    _lib.srukf_set_measurement.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_measurement.restype = None
    
    # srukf_set_process_noise(ukf, Sq)
    _lib.srukf_set_process_noise.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_process_noise.restype = None
    
    # srukf_set_measurement_noise(ukf, Sr)
    _lib.srukf_set_measurement_noise.argtypes = [c_void_p, c_double_p]
    _lib.srukf_set_measurement_noise.restype = None
    
    # srukf_predict(ukf)
    _lib.srukf_predict.argtypes = [c_void_p]
    _lib.srukf_predict.restype = None
    
    # srukf_update(ukf, z)
    _lib.srukf_update.argtypes = [c_void_p, c_double_p]
    _lib.srukf_update.restype = None
    
    # srukf_step(ukf, z)
    _lib.srukf_step.argtypes = [c_void_p, c_double_p]
    _lib.srukf_step.restype = None
    
    # srukf_get_state(ukf) -> const double*
    _lib.srukf_get_state.argtypes = [c_void_p]
    _lib.srukf_get_state.restype = c_double_p
    
    # srukf_get_sqrt_cov(ukf) -> const double*
    _lib.srukf_get_sqrt_cov.argtypes = [c_void_p]
    _lib.srukf_get_sqrt_cov.restype = c_double_p
    
    # srukf_get_nis(ukf) -> double
    _lib.srukf_get_nis.argtypes = [c_void_p]
    _lib.srukf_get_nis.restype = ctypes.c_double
    
    # srukf_get_innovation(ukf) -> const double*
    _lib.srukf_get_innovation.argtypes = [c_void_p]
    _lib.srukf_get_innovation.restype = c_double_p
    
    # srukf_enable_nis_tracking(ukf, window_size, threshold)
    _lib.srukf_enable_nis_tracking.argtypes = [c_void_p, ctypes.c_int, ctypes.c_double]
    _lib.srukf_enable_nis_tracking.restype = None
    
    # srukf_nis_healthy(ukf) -> bool
    _lib.srukf_nis_healthy.argtypes = [c_void_p]
    _lib.srukf_nis_healthy.restype = ctypes.c_bool
    
    # srukf_reset(ukf, x0, S0)
    _lib.srukf_reset.argtypes = [c_void_p, c_double_p, c_double_p]
    _lib.srukf_reset.restype = None
    
    # srukf_predict_only(ukf)
    _lib.srukf_predict_only.argtypes = [c_void_p]
    _lib.srukf_predict_only.restype = None
    
    # srukf_check_covariance(ukf) -> bool
    _lib.srukf_check_covariance.argtypes = [c_void_p]
    _lib.srukf_check_covariance.restype = ctypes.c_bool
    
    # srukf_repair_covariance(ukf, min_diag) -> bool
    _lib.srukf_repair_covariance.argtypes = [c_void_p, ctypes.c_double]
    _lib.srukf_repair_covariance.restype = ctypes.c_bool

_setup_signatures()

# =============================================================================
# Helper Functions
# =============================================================================

def _to_c_array(arr: np.ndarray, dtype=np.float64) -> c_double_p:
    """Convert numpy array to ctypes pointer."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr.ctypes.data_as(c_double_p)

def _to_fortran_array(arr: np.ndarray) -> c_double_p:
    """Convert numpy array to column-major (Fortran) ctypes pointer."""
    arr = np.asfortranarray(arr, dtype=np.float64)
    return arr.ctypes.data_as(c_double_p)

def _from_c_array(ptr: c_double_p, shape: tuple) -> np.ndarray:
    """Create numpy array from ctypes pointer (copy)."""
    size = int(np.prod(shape))
    arr = np.ctypeslib.as_array(ptr, shape=(size,)).copy()
    return arr.reshape(shape, order='F') if len(shape) > 1 else arr

# =============================================================================
# NIS Statistics Structure
# =============================================================================

class SRUKF_NIS_Stats(ctypes.Structure):
    """NIS statistics structure matching C struct."""
    _fields_ = [
        ("mean", ctypes.c_double),
        ("variance", ctypes.c_double),
        ("trend", ctypes.c_double),
        ("max_recent", ctypes.c_double),
        ("n_outliers", ctypes.c_int),
        ("window_fill", ctypes.c_int),
    ]

# Set up get_nis_stats signature
_lib.srukf_get_nis_stats.argtypes = [c_void_p, ctypes.POINTER(SRUKF_NIS_Stats)]
_lib.srukf_get_nis_stats.restype = None

# =============================================================================
# Student-t Square-Root UKF
# =============================================================================

@dataclass
class NISStats:
    """NIS statistics for model health monitoring."""
    mean: float
    variance: float
    trend: float
    max_recent: float
    n_outliers: int
    window_fill: int
    
    @property
    def fraction_above(self) -> float:
        """Fraction of NIS values above threshold."""
        if self.window_fill == 0:
            return 0.0
        return self.n_outliers / self.window_fill
    
    @property
    def healthy(self) -> bool:
        """Quick health check (< 30% outliers)."""
        return self.fraction_above < 0.3


class StudentTSRUKF:
    """
    Student-t Square-Root Unscented Kalman Filter.
    
    Robust state estimation with:
    - Square-root covariance (numerical stability)
    - Student-t likelihood (outlier robustness)
    - MKL-accelerated linear algebra (~2μs per step)
    
    Parameters
    ----------
    nx : int
        State dimension
    nz : int
        Measurement dimension
    nu : float
        Student-t degrees of freedom (4-6 typical, inf for Gaussian)
    
    Example
    -------
    >>> ukf = StudentTSRUKF(nx=3, nz=1, nu=4.0)
    >>> ukf.set_state([100.0, 0.0, -3.0])  # level, velocity, log_vol
    >>> ukf.set_dynamics(np.eye(3))
    >>> ukf.set_measurement(np.array([[1, 0, 0]]))
    >>> ukf.set_process_noise(np.diag([0.1, 0.01, 0.05]))
    >>> ukf.set_measurement_noise(np.array([[1.0]]))
    >>> 
    >>> for z in measurements:
    ...     ukf.step(z)
    ...     print(f"State: {ukf.state}, NIS: {ukf.nis:.2f}")
    """
    
    def __init__(self, nx: int, nz: int, nu: float = 4.0):
        self.nx = nx
        self.nz = nz
        self.nu = nu
        
        self._ptr = _lib.srukf_create(nx, nz, ctypes.c_double(nu))
        if not self._ptr:
            raise MemoryError("Failed to create UKF")
        
        self._owns_ptr = True
    
    def __del__(self):
        if hasattr(self, '_owns_ptr') and self._owns_ptr and self._ptr:
            _lib.srukf_destroy(self._ptr)
            self._ptr = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self._owns_ptr and self._ptr:
            _lib.srukf_destroy(self._ptr)
            self._ptr = None
            self._owns_ptr = False
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    def set_state(self, x: np.ndarray):
        """Set state vector."""
        x = np.asarray(x, dtype=np.float64).ravel()
        if len(x) != self.nx:
            raise ValueError(f"Expected {self.nx} elements, got {len(x)}")
        _lib.srukf_set_state(self._ptr, _to_c_array(x))
    
    def set_sqrt_cov(self, S: np.ndarray):
        """Set sqrt covariance (lower triangular Cholesky factor)."""
        S = np.asarray(S, dtype=np.float64)
        if S.shape != (self.nx, self.nx):
            raise ValueError(f"Expected ({self.nx}, {self.nx}), got {S.shape}")
        _lib.srukf_set_sqrt_cov(self._ptr, _to_fortran_array(S))
    
    def set_covariance(self, P: np.ndarray):
        """Set covariance (computes Cholesky internally)."""
        P = np.asarray(P, dtype=np.float64)
        if P.shape != (self.nx, self.nx):
            raise ValueError(f"Expected ({self.nx}, {self.nx}), got {P.shape}")
        S = np.linalg.cholesky(P)
        self.set_sqrt_cov(S)
    
    def set_dynamics(self, F: np.ndarray):
        """Set state transition matrix."""
        F = np.asarray(F, dtype=np.float64)
        if F.shape != (self.nx, self.nx):
            raise ValueError(f"Expected ({self.nx}, {self.nx}), got {F.shape}")
        _lib.srukf_set_dynamics(self._ptr, _to_fortran_array(F))
    
    def set_measurement(self, H: np.ndarray):
        """Set measurement matrix."""
        H = np.asarray(H, dtype=np.float64)
        if H.shape != (self.nz, self.nx):
            raise ValueError(f"Expected ({self.nz}, {self.nx}), got {H.shape}")
        _lib.srukf_set_measurement(self._ptr, _to_fortran_array(H))
    
    def set_process_noise(self, Q_or_Sq: np.ndarray, is_sqrt: bool = True):
        """
        Set process noise.
        
        Parameters
        ----------
        Q_or_Sq : ndarray
            If is_sqrt=True: sqrt of Q (lower triangular)
            If is_sqrt=False: Q itself (computes Cholesky)
        is_sqrt : bool
            Whether input is already sqrt form
        """
        Q_or_Sq = np.asarray(Q_or_Sq, dtype=np.float64)
        if Q_or_Sq.shape != (self.nx, self.nx):
            raise ValueError(f"Expected ({self.nx}, {self.nx}), got {Q_or_Sq.shape}")
        
        if not is_sqrt:
            Q_or_Sq = np.linalg.cholesky(Q_or_Sq)
        
        _lib.srukf_set_process_noise(self._ptr, _to_fortran_array(Q_or_Sq))
    
    def set_measurement_noise(self, R_or_Sr: np.ndarray, is_sqrt: bool = True):
        """
        Set measurement noise.
        
        Parameters
        ----------
        R_or_Sr : ndarray
            If is_sqrt=True: sqrt of R (lower triangular)
            If is_sqrt=False: R itself (computes Cholesky)
        is_sqrt : bool
            Whether input is already sqrt form
        """
        R_or_Sr = np.asarray(R_or_Sr, dtype=np.float64)
        if R_or_Sr.shape != (self.nz, self.nz):
            raise ValueError(f"Expected ({self.nz}, {self.nz}), got {R_or_Sr.shape}")
        
        if not is_sqrt:
            R_or_Sr = np.linalg.cholesky(R_or_Sr)
        
        _lib.srukf_set_measurement_noise(self._ptr, _to_fortran_array(R_or_Sr))
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def predict(self):
        """Run predict step only."""
        _lib.srukf_predict(self._ptr)
    
    def update(self, z: np.ndarray):
        """Run update step only."""
        z = np.asarray(z, dtype=np.float64).ravel()
        if len(z) != self.nz:
            raise ValueError(f"Expected {self.nz} measurements, got {len(z)}")
        _lib.srukf_update(self._ptr, _to_c_array(z))
    
    def step(self, z: np.ndarray):
        """Run full predict + update step."""
        z = np.asarray(z, dtype=np.float64).ravel()
        if len(z) != self.nz:
            raise ValueError(f"Expected {self.nz} measurements, got {len(z)}")
        _lib.srukf_step(self._ptr, _to_c_array(z))
    
    def predict_only(self):
        """Predict without update (missing measurement)."""
        _lib.srukf_predict_only(self._ptr)
    
    def reset(self, x0: np.ndarray, S0: np.ndarray):
        """
        Reset filter to specified state.
        
        Parameters
        ----------
        x0 : ndarray
            Initial state (nx,)
        S0 : ndarray
            Initial sqrt covariance (nx, nx)
        """
        x0 = np.asarray(x0, dtype=np.float64).ravel()
        S0 = np.asarray(S0, dtype=np.float64)
        if len(x0) != self.nx:
            raise ValueError(f"Expected {self.nx} elements, got {len(x0)}")
        if S0.shape != (self.nx, self.nx):
            raise ValueError(f"Expected ({self.nx}, {self.nx}), got {S0.shape}")
        _lib.srukf_reset(self._ptr, _to_c_array(x0), _to_fortran_array(S0))
    
    # -------------------------------------------------------------------------
    # State Access
    # -------------------------------------------------------------------------
    
    @property
    def state(self) -> np.ndarray:
        """Current state estimate."""
        ptr = _lib.srukf_get_state(self._ptr)
        return _from_c_array(ptr, (self.nx,))
    
    @property
    def x(self) -> np.ndarray:
        """Alias for state."""
        return self.state
    
    @property
    def sqrt_cov(self) -> np.ndarray:
        """Current sqrt covariance (lower triangular)."""
        ptr = _lib.srukf_get_sqrt_cov(self._ptr)
        return _from_c_array(ptr, (self.nx, self.nx))
    
    @property
    def S(self) -> np.ndarray:
        """Alias for sqrt_cov."""
        return self.sqrt_cov
    
    @property
    def covariance(self) -> np.ndarray:
        """Current covariance P = S @ S.T."""
        S = self.sqrt_cov
        return S @ S.T
    
    @property
    def P(self) -> np.ndarray:
        """Alias for covariance."""
        return self.covariance
    
    @property
    def nis(self) -> float:
        """Normalized Innovation Squared from last update."""
        return _lib.srukf_get_nis(self._ptr)
    
    @property
    def innovation(self) -> np.ndarray:
        """Innovation (residual) from last update."""
        ptr = _lib.srukf_get_innovation(self._ptr)
        return _from_c_array(ptr, (self.nz,))
    
    # -------------------------------------------------------------------------
    # Health Monitoring
    # -------------------------------------------------------------------------
    
    def enable_nis_tracking(self, window_size: int = 50, threshold: float = None):
        """
        Enable windowed NIS tracking for model health.
        
        Parameters
        ----------
        window_size : int
            Rolling window size
        threshold : float
            NIS threshold for outlier counting (default: 3 * nz)
        """
        if threshold is None:
            threshold = 3.0 * self.nz
        _lib.srukf_enable_nis_tracking(self._ptr, window_size, ctypes.c_double(threshold))
    
    def get_nis_stats(self) -> NISStats:
        """Get current NIS statistics."""
        stats = SRUKF_NIS_Stats()
        _lib.srukf_get_nis_stats(self._ptr, ctypes.byref(stats))
        return NISStats(
            mean=stats.mean,
            variance=stats.variance,
            trend=stats.trend,
            max_recent=stats.max_recent,
            n_outliers=stats.n_outliers,
            window_fill=stats.window_fill,
        )
    
    @property
    def healthy(self) -> bool:
        """Quick model health check."""
        return _lib.srukf_nis_healthy(self._ptr)
    
    def check_cov_health(self) -> bool:
        """Check covariance health (True = healthy)."""
        return _lib.srukf_check_covariance(self._ptr)
    
    def repair_cov(self, min_diag: float = 1e-6) -> bool:
        """Repair covariance if unhealthy. Returns True if repair was needed."""
        return _lib.srukf_repair_covariance(self._ptr, ctypes.c_double(min_diag))
    
    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------
    
    def filter(self, measurements: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run filter on batch of measurements.
        
        Parameters
        ----------
        measurements : ndarray
            Array of shape (T, nz) or (T,) if nz=1
        
        Returns
        -------
        states : ndarray
            Filtered states (T, nx)
        covariances : ndarray
            Covariances (T, nx, nx)
        nis_values : ndarray
            NIS values (T,)
        """
        measurements = np.atleast_2d(measurements)
        if measurements.ndim == 1:
            measurements = measurements.reshape(-1, 1)
        
        T = len(measurements)
        states = np.zeros((T, self.nx))
        covariances = np.zeros((T, self.nx, self.nx))
        nis_values = np.zeros(T)
        
        for t in range(T):
            self.step(measurements[t])
            states[t] = self.state
            covariances[t] = self.covariance
            nis_values[t] = self.nis
        
        return states, covariances, nis_values


# =============================================================================
# Kelly Criterion
# =============================================================================

@dataclass
class KellyResult:
    """Kelly criterion calculation result."""
    f_full: float           # Full Kelly fraction
    f_half: float           # Half Kelly
    f_adjusted: float       # After uncertainty/tail adjustment
    f_final: float          # After position limits
    
    expected_return: float  # μ
    volatility: float       # σ
    sharpe: float           # μ/σ
    
    mu_uncertainty: float   # Std of μ estimate
    vol_uncertainty: float  # Std of log-vol estimate
    
    capped_long: bool       # Hit max leverage?
    capped_short: bool      # Hit min leverage?
    
    @property
    def position(self) -> float:
        """Final position size (alias for f_final)."""
        return self.f_final


class Kelly:
    """
    Kelly Criterion calculator with UKF integration.
    
    Features:
    - Bayesian variance from log-volatility uncertainty
    - Student-t tail adjustment
    - Weak signal filtering
    - Asymmetric short penalty
    - Transaction cost aware
    """
    
    # Configuration (match C header)
    MAX_LEVERAGE = 2.0
    MIN_LEVERAGE = -1.0
    MIN_VOLATILITY = 1e-6
    MIN_SIGNAL_RATIO = 1.0
    SHORT_PENALTY = 0.5
    TAIL_FLOOR_NU = 2.1
    
    @staticmethod
    def simple(mu: float, sigma: float, nu: float = np.inf, fraction: float = 0.5) -> float:
        """
        Simple Kelly: f* = μ / σ²
        
        Parameters
        ----------
        mu : float
            Expected return
        sigma : float
            Volatility
        nu : float
            Student-t degrees of freedom (inf for Gaussian)
        fraction : float
            Kelly fraction (0.5 = half Kelly)
        
        Returns
        -------
        float
            Position size
        """
        if sigma < Kelly.MIN_VOLATILITY:
            sigma = Kelly.MIN_VOLATILITY
        
        var = sigma ** 2
        
        # Tail adjustment
        if np.isfinite(nu):
            if nu <= Kelly.TAIL_FLOOR_NU:
                var *= 10.0
            else:
                var *= nu / (nu - 2.0)
        
        f = fraction * mu / var
        
        # Short penalty
        if f < 0:
            f *= Kelly.SHORT_PENALTY
        
        # Clamp
        f = np.clip(f, Kelly.MIN_LEVERAGE, Kelly.MAX_LEVERAGE)
        
        return f
    
    @staticmethod
    def _extract_std(S: np.ndarray, idx: int) -> float:
        """Extract std from Cholesky factor (row sum of squares)."""
        # S is lower triangular, Var(x_i) = sum_{k=0}^{i} S[i,k]²
        return np.sqrt(np.sum(S[idx, :idx+1] ** 2))
    
    @staticmethod
    def _expected_variance(mu_lv: float, sigma_lv: float) -> float:
        """Bayesian E[σ²] from log-volatility distribution."""
        return np.exp(2.0 * mu_lv + 2.0 * sigma_lv ** 2)
    
    @staticmethod
    def _tail_variance(base_var: float, nu: float) -> float:
        """Tail-adjusted variance for Student-t."""
        if not np.isfinite(nu):
            return base_var
        if nu <= Kelly.TAIL_FLOOR_NU:
            return base_var * 10.0
        return base_var * nu / (nu - 2.0)
    
    @staticmethod
    def from_ukf(
        ukf: StudentTSRUKF,
        vel_idx: int = 1,
        vol_idx: int = 2,
        fraction: float = 0.5,
    ) -> KellyResult:
        """
        Compute Kelly from UKF state.
        
        Parameters
        ----------
        ukf : StudentTSRUKF
            Filter instance
        vel_idx : int
            Index of velocity (expected return) in state
        vol_idx : int
            Index of log-volatility in state
        fraction : float
            Kelly fraction
        
        Returns
        -------
        KellyResult
            Full Kelly calculation result
        """
        x = ukf.state
        S = ukf.sqrt_cov
        nu = ukf.nu
        
        return Kelly.from_state(x, S, vel_idx, vol_idx, nu, fraction)
    
    @staticmethod
    def from_state(
        x: np.ndarray,
        S: np.ndarray,
        vel_idx: int = 1,
        vol_idx: int = 2,
        nu: float = np.inf,
        fraction: float = 0.5,
    ) -> KellyResult:
        """
        Compute Kelly from state and sqrt covariance.
        
        Parameters
        ----------
        x : ndarray
            State vector
        S : ndarray
            Sqrt covariance (lower triangular)
        vel_idx : int
            Index of velocity in state
        vol_idx : int
            Index of log-volatility in state
        nu : float
            Student-t degrees of freedom
        fraction : float
            Kelly fraction
        
        Returns
        -------
        KellyResult
            Full Kelly calculation result
        """
        # Extract estimates
        mu = x[vel_idx]
        mu_lv = x[vol_idx]
        
        # Extract uncertainties
        sigma_mu = Kelly._extract_std(S, vel_idx)
        sigma_lv = Kelly._extract_std(S, vol_idx)
        
        # Point estimate of volatility
        sigma_point = np.exp(mu_lv)
        
        # Bayesian expected variance
        expected_var = Kelly._expected_variance(mu_lv, sigma_lv)
        
        # Tail adjustment
        tail_var = Kelly._tail_variance(expected_var, nu)
        
        # Floor variance
        if tail_var < Kelly.MIN_VOLATILITY ** 2:
            tail_var = Kelly.MIN_VOLATILITY ** 2
        
        # Weak signal filter
        signal_strength = abs(mu) / sigma_mu if sigma_mu > 0 else np.inf
        
        if signal_strength < Kelly.MIN_SIGNAL_RATIO:
            # Signal too weak
            return KellyResult(
                f_full=0.0,
                f_half=0.0,
                f_adjusted=0.0,
                f_final=0.0,
                expected_return=mu,
                volatility=sigma_point,
                sharpe=mu / sigma_point if sigma_point > 0 else 0.0,
                mu_uncertainty=sigma_mu,
                vol_uncertainty=sigma_lv,
                capped_long=False,
                capped_short=False,
            )
        
        # Bayesian Kelly
        f_full = mu / tail_var
        f_half = f_full * 0.5
        f_adjusted = f_full * fraction
        
        # Asymmetric short penalty
        if f_adjusted < 0:
            f_adjusted *= Kelly.SHORT_PENALTY
        
        # Position limits
        f_final = f_adjusted
        capped_long = False
        capped_short = False
        
        if f_final > Kelly.MAX_LEVERAGE:
            f_final = Kelly.MAX_LEVERAGE
            capped_long = True
        elif f_final < Kelly.MIN_LEVERAGE:
            f_final = Kelly.MIN_LEVERAGE
            capped_short = True
        
        return KellyResult(
            f_full=f_full,
            f_half=f_half,
            f_adjusted=f_adjusted,
            f_final=f_final,
            expected_return=mu,
            volatility=sigma_point,
            sharpe=mu / sigma_point if sigma_point > 0 else 0.0,
            mu_uncertainty=sigma_mu,
            vol_uncertainty=sigma_lv,
            capped_long=capped_long,
            capped_short=capped_short,
        )
    
    @staticmethod
    def growth_rate(mu: float, sigma: float, f: float) -> float:
        """Expected log growth rate: g(f) = μf - ½σ²f²"""
        return mu * f - 0.5 * sigma ** 2 * f ** 2
    
    @staticmethod
    def optimal_growth(mu: float, sigma: float) -> float:
        """Optimal growth rate at full Kelly: g* = μ²/(2σ²)"""
        if sigma < Kelly.MIN_VOLATILITY:
            sigma = Kelly.MIN_VOLATILITY
        return 0.5 * (mu / sigma) ** 2
    
    @staticmethod
    def health_scale(nis_above: float, soft: float = 0.1, hard: float = 0.3) -> float:
        """
        Graduated position scaling based on model health.
        
        Parameters
        ----------
        nis_above : float
            Fraction of NIS above threshold (0-1)
        soft : float
            Start reducing at this level
        hard : float
            Zero position at this level
        
        Returns
        -------
        float
            Scale factor (0-1)
        """
        if nis_above <= soft:
            return 1.0
        if nis_above >= hard:
            return 0.0
        return (hard - nis_above) / (hard - soft)


# =============================================================================
# Convenience Functions
# =============================================================================

def create_trend_filter(nu: float = 4.0) -> StudentTSRUKF:
    """
    Create a standard trend-following filter.
    
    State: [level, velocity, log_vol]
    Measurement: level
    
    Parameters
    ----------
    nu : float
        Student-t degrees of freedom
    
    Returns
    -------
    StudentTSRUKF
        Configured filter
    """
    ukf = StudentTSRUKF(nx=3, nz=1, nu=nu)
    
    # State transition: random walk + velocity + mean-reverting vol
    F = np.array([
        [1.0, 1.0, 0.0],   # level += velocity
        [0.0, 1.0, 0.0],   # velocity persists
        [0.0, 0.0, 0.95],  # log_vol mean-reverts
    ])
    ukf.set_dynamics(F)
    
    # Observe level only
    H = np.array([[1.0, 0.0, 0.0]])
    ukf.set_measurement(H)
    
    # Process noise
    Sq = np.diag([0.1, 0.01, 0.05])
    ukf.set_process_noise(Sq)
    
    # Measurement noise
    Sr = np.array([[1.0]])
    ukf.set_measurement_noise(Sr)
    
    # Initial state
    ukf.set_state([0.0, 0.0, -3.0])  # log_vol = -3 → σ ≈ 5%
    ukf.set_sqrt_cov(np.diag([1.0, 0.1, 0.1]))
    
    return ukf


# =============================================================================
# Main (Demo)
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("Student-t SQR UKF - Python Bindings Demo")
    print("=" * 60)
    
    # Create filter
    ukf = create_trend_filter(nu=4.0)
    ukf.enable_nis_tracking(window_size=50)
    
    # Generate synthetic data
    np.random.seed(42)
    T = 1000
    true_level = 100.0
    true_velocity = 0.1
    true_vol = 0.05
    
    measurements = np.zeros(T)
    for t in range(T):
        true_level += true_velocity + np.random.standard_t(4) * true_vol
        measurements[t] = true_level + np.random.randn() * 1.0
    
    # Run filter
    print(f"\nFiltering {T} measurements...")
    
    start = time.perf_counter()
    states, covs, nis_vals = ukf.filter(measurements)
    elapsed = time.perf_counter() - start
    
    print(f"Time: {elapsed*1000:.2f} ms ({elapsed/T*1e6:.2f} μs/step)")
    
    # Final state
    print(f"\nFinal state:")
    print(f"  Level:    {ukf.state[0]:.2f}")
    print(f"  Velocity: {ukf.state[1]:.4f}")
    print(f"  Log-vol:  {ukf.state[2]:.2f} (σ = {np.exp(ukf.state[2]):.4f})")
    
    # Kelly
    kelly = Kelly.from_ukf(ukf, fraction=0.5)
    print(f"\nKelly position: {kelly.position:.4f}")
    print(f"  Sharpe: {kelly.sharpe:.2f}")
    print(f"  μ uncertainty: {kelly.mu_uncertainty:.4f}")
    
    # NIS health
    stats = ukf.get_nis_stats()
    print(f"\nModel health:")
    print(f"  NIS mean: {stats.mean:.2f} (expected: 1.0)")
    print(f"  Outliers: {stats.fraction_above*100:.1f}%")
    print(f"  Healthy: {stats.healthy}")
    
    print("\n" + "=" * 60)