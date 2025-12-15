# Student-t Square-Root Unscented Kalman Filter

A production-grade, MKL-optimized implementation designed for quantitative trading systems.

## Table of Contents

1. [Motivation](#motivation)
2. [Architecture](#architecture)
3. [Mathematical Foundation](#mathematical-foundation)
4. [Design Decisions](#design-decisions)
5. [Implementation Details](#implementation-details)
6. [Features](#features)
7. [API Reference](#api-reference)
8. [Usage Examples](#usage-examples)
9. [Performance](#performance)
10. [Integration](#integration)

---

## Motivation

### The Problem

Standard Kalman filters assume Gaussian noise. Financial markets exhibit:

- **Fat tails**: Returns are roughly Student-t(4) distributed, not Gaussian
- **Outliers**: Flash crashes, news events, gaps create extreme observations
- **State-dependent volatility**: Noise varies with market regime

A standard UKF gets "yanked around" by outliers, producing unstable estimates.

### The Solution

Combine three orthogonal techniques:

| Technique | Problem Solved |
|-----------|----------------|
| **UKF** (Unscented) | Nonlinear state-dependent noise |
| **Square-Root** | Numerical stability |
| **Student-t weighting** | Fat-tail robustness |

The result: a filter that tracks state accurately while automatically downweighting outliers.

---

The UKF deliberately uses a simple model because:
- SSA already extracted structure
- BOCPD detects when dynamics change
- Within a regime, simple linear dynamics suffice

---

## Mathematical Foundation

### State Model

**State vector:**
```
x = [α, α̇, ξ]ᵀ

where:
  α  = trend level (price or return level)
  α̇  = trend velocity (momentum)
  ξ  = log(σ) (log-volatility)
```

**Dynamics (linear):**
```
α(t+1)  = α(t) + Δt·α̇(t) + w_α
α̇(t+1) = φ·α̇(t) + w_α̇           (AR(1) mean reversion)
ξ(t+1)  = μ_ξ + ψ·(ξ(t) - μ_ξ) + w_ξ   (mean-reverting log-vol)
```

**Measurement:**
```
z = H·x + v

where v ~ N(0, R(x)) and R(x) = exp(2ξ)·R₀
```

The key insight: nonlinearity is concentrated in the measurement noise scaling via `exp(ξ)`. State dynamics remain linear, making this a minimal-complexity model that still requires UKF (for the state-dependent noise).

### UKF Sigma Points

Generate 2n+1 sigma points:
```
X₀ = x̄
Xᵢ = x̄ + γ·Sᵢ      for i = 1,...,n
Xᵢ = x̄ - γ·Sᵢ₋ₙ    for i = n+1,...,2n

where:
  γ = √(n + λ)
  λ = α²(n + κ) - n
  S = sqrt covariance (Cholesky factor)
```

### Student-t Weighting

Standard update:
```
x⁺ = x⁻ + K·ν
```

Student-t modification:
```
x⁺ = x⁻ + w·K·ν

where:
  d² = ν'·Sᵧᵧ⁻¹·ν    (Mahalanobis distance)
  w  = (ν + nz) / (ν + d²)
```

**Effect:**

| Observation | d² | w | Action |
|-------------|-----|-----|--------|
| Normal | ~1 | ~1.0 | Full update |
| 3σ outlier | ~9 | ~0.4 | Partial update |
| 5σ outlier | ~25 | ~0.17 | Heavily downweighted |

The filter automatically becomes skeptical of extreme observations.

### Square-Root Form

Instead of maintaining P (covariance), maintain S where P = S·Sᵀ.

**Benefits:**
- Guaranteed positive semi-definite
- Half the condition number
- Better numerical stability
- Natural for Cholesky-based updates

**Covariance update via rank-1 downdate:**
```
S_new·S_new' = S·S' - w·(K·Sᵧᵧ)·(K·Sᵧᵧ)'
```

Computed via Givens rotations, not explicit matrix operations.

---

## Design Decisions

### Why MKL?

| Alternative | Problem |
|-------------|---------|
| Hand-rolled AVX512 | Maintenance burden, hardware-specific |
| OpenBLAS | Good, but MKL faster on Intel |
| Eigen | C++, not C |

MKL provides:
- Automatic SIMD dispatch (AVX512/AVX2/SSE)
- Multi-threaded primitives (optional)
- Battle-tested numerical accuracy
- Free performance upgrades with hardware

### Why BLAS 3?

BLAS levels by compute intensity:

| Level | Operations | Compute/Memory |
|-------|------------|----------------|
| BLAS 1 | Vector-vector | O(1) |
| BLAS 2 | Matrix-vector | O(n) |
| BLAS 3 | Matrix-matrix | O(n²) |

BLAS 3 keeps the CPU fed. Key operations:
- `dgemm`: Sigma point propagation
- `dsyrk`: Covariance from centered points
- `dtrsm`: Kalman gain via triangular solve

### Why Single Allocation?

```c
// Bad: Multiple allocations
ukf->x = malloc(nx * sizeof(double));
ukf->S = malloc(nx * nx * sizeof(double));
// ... 20 more allocations

// Good: Single contiguous block
ukf->_memory = mkl_malloc(total_size, 64);
ukf->x = (double*)(base + off_x);
ukf->S = (double*)(base + off_S);
```

Benefits:
- One allocation at init, zero in hot path
- Better cache locality
- Simpler cleanup
- 64-byte alignment for AVX512

### Why Not Dynamic ν Estimation?

Considered but rejected:

```
ν_new = ν_old + η·((d²/nz) - 1)
```

Problems:
1. Sign is backwards (large d² should decrease ν)
2. BOCPD already detects regime changes
3. Student-t weight w adapts per-observation
4. Adds failure mode without clear benefit

If ν needs to change, do it at BOCPD changepoints, not continuously.

### Why Not Adaptive Q?

Considered:
```
Q_t = Q_base + c·innovation²
```

Problems:
1. Conflicts with Student-t weight (w downweights, Q amplifies)
2. Creates feedback loops
3. BOCPD is the right layer for detecting model mismatch

---

## Implementation Details

### Memory Layout

```
┌──────────────────────────────────────────────────────────┐
│                 Single Aligned Block                      │
├──────────┬───────┬───────┬───────┬───────┬──────────────┤
│ x [nx]   │S[nx²] │F[nx²] │H[nz×nx]│ Sq    │ ...         │
├──────────┴───────┴───────┴───────┴───────┴──────────────┤
│ X_sig    │X_pred │Z_sig  │ Wm    │ Wc    │Wc_sqrt│ ... │
├──────────┴───────┴───────┴───────┴───────┴───────┴──────┤
│ x_pred   │z_pred │innov  │ Pxz   │ K     │ Szz   │ ... │
├──────────┴───────┴───────┴───────┴───────┴───────┴──────┤
│ xi_work  │R_scales│work_mat│c_work│s_work │              │
└──────────────────────────────────────────────────────────┘
        All 64-byte aligned for AVX512
```

### Optimizations Applied

1. **Reuse sigma points**: Predict generates X_pred, update consumes it directly
2. **Precompute Wc_sqrt**: sqrt(|Wc|)·sign(Wc) computed once in set_params
3. **Fused loops**: Z_centered and Z_c computed in single pass
4. **Vectorized exp**: `vdExp()` for state-dependent noise scaling
5. **No QR**: Switched to Cholesky + rank-1 downdates

### Column-Major Convention

All matrices are column-major (Fortran/MKL convention):

```c
// Matrix F where F[i,j] = F[i + j*nx]
// 
// F = | 1   dt  0  |
//     | 0   φ   0  |
//     | 0   0   ψ  |
//
double F[9] = {
    1.0, 0.0, 0.0,    // Column 0
    dt,  phi, 0.0,    // Column 1
    0.0, 0.0, psi     // Column 2
};
```

---

## Features

### Core Filtering

| Function | Description |
|----------|-------------|
| `srukf_predict()` | Propagate state through dynamics |
| `srukf_update()` | Incorporate measurement with Student-t |
| `srukf_step()` | Combined predict + update |

### Missing Data Handling

Markets close. Data has gaps. The filter handles this:

```c
// No observation available (holiday, gap)
srukf_predict_only(ukf);  // State propagates, uncertainty grows

// Partial observation (some sensors failed)
bool mask[3] = {true, false, true};  // Only components 0 and 2
srukf_update_partial(ukf, z, mask);
```

### Covariance Health

Numerical issues happen. Detect and repair:

```c
// Check health
if (!srukf_check_covariance(ukf)) {
    srukf_repair_covariance(ukf, 1e-8);  // Enforce minimum diagonal
}

// Check for blow-up
if (!srukf_check_state_bounds(ukf, 1e6)) {
    // State exploded - reset or halt
}
```

### NIS Statistics for Kill Switch

Track filter health over time:

```c
// Enable tracking
srukf_enable_nis_tracking(ukf, 50, 3.0);  // window=50, threshold=3*nz

// After each update
SRUKF_NIS_Stats stats;
srukf_get_nis_stats(ukf, &stats);

// stats.mean       - should be ~nz if model correct
// stats.variance   - stability indicator
// stats.trend      - positive = degrading
// stats.n_outliers - outlier rate
// stats.max_recent - worst case in window

// Quick health check
if (!srukf_nis_healthy(ukf)) {
    reduce_position();  // Model degrading
}
```

### Serialization

Save/restore state across restarts:

```c
// Save state
size_t size = srukf_serialize_size(ukf);
void* buf = malloc(size);
srukf_serialize(ukf, buf, size);
write_to_file(buf, size);

// Later: restore
read_from_file(buf, size);
srukf_deserialize(ukf, buf, size);
// Filter continues from saved state
```

Note: Model matrices (F, H, Q, R) are not serialized - assumed constant.

### State Reset

Reinitialize without reallocating:

```c
srukf_reset(ukf, x0, S0);  // New state, clears history
```

---

## API Reference

### Lifecycle

```c
StudentT_SRUKF* srukf_create(int nx, int nz, double nu);
void srukf_destroy(StudentT_SRUKF* ukf);
```

### Configuration

```c
void srukf_set_state(StudentT_SRUKF* ukf, const double* x0);
void srukf_set_sqrt_cov(StudentT_SRUKF* ukf, const double* S0);
void srukf_set_dynamics(StudentT_SRUKF* ukf, const double* F);
void srukf_set_measurement(StudentT_SRUKF* ukf, const double* H);
void srukf_set_process_noise(StudentT_SRUKF* ukf, const double* Sq);
void srukf_set_measurement_noise(StudentT_SRUKF* ukf, const double* R0);
void srukf_set_ukf_params(StudentT_SRUKF* ukf, double alpha, double beta, double kappa);
void srukf_set_vol_index(StudentT_SRUKF* ukf, int xi_index);
void srukf_set_student_nu(StudentT_SRUKF* ukf, double nu);
```

### Runtime (Hot Path)

```c
void srukf_predict(StudentT_SRUKF* ukf);
void srukf_update(StudentT_SRUKF* ukf, const double* z);
void srukf_step(StudentT_SRUKF* ukf, const double* z);
void srukf_predict_only(StudentT_SRUKF* ukf);
int srukf_update_partial(StudentT_SRUKF* ukf, const double* z, const bool* mask);
```

### Accessors

```c
const double* srukf_get_state(const StudentT_SRUKF* ukf);
const double* srukf_get_sqrt_cov(const StudentT_SRUKF* ukf);
double srukf_get_nis(const StudentT_SRUKF* ukf);
double srukf_get_student_weight(const StudentT_SRUKF* ukf);
double srukf_get_volatility(const StudentT_SRUKF* ukf);
double srukf_get_mahalanobis_sq(const StudentT_SRUKF* ukf);
int srukf_get_nx(const StudentT_SRUKF* ukf);
int srukf_get_nz(const StudentT_SRUKF* ukf);
```

### Health & Diagnostics

```c
bool srukf_check_covariance(const StudentT_SRUKF* ukf);
bool srukf_repair_covariance(StudentT_SRUKF* ukf, double min_diag);
bool srukf_check_state_bounds(const StudentT_SRUKF* ukf, double max_abs);
void srukf_enable_nis_tracking(StudentT_SRUKF* ukf, int window_size, double threshold);
void srukf_get_nis_stats(const StudentT_SRUKF* ukf, SRUKF_NIS_Stats* stats);
bool srukf_nis_healthy(const StudentT_SRUKF* ukf);
```

### Serialization

```c
void srukf_reset(StudentT_SRUKF* ukf, const double* x0, const double* S0);
size_t srukf_serialize_size(const StudentT_SRUKF* ukf);
size_t srukf_serialize(const StudentT_SRUKF* ukf, void* buf, size_t size);
bool srukf_deserialize(StudentT_SRUKF* ukf, const void* buf, size_t size);
uint32_t srukf_version(void);
```

---

## Usage Examples

### Basic Usage

```c
#include "student_t_srukf.h"

int main() {
    // Create filter: 3 states, 1 measurement, nu=4
    StudentT_SRUKF* ukf = srukf_create(3, 1, 4.0);
    
    // Initialize state: [trend=0, velocity=0, log_vol=log(0.02)]
    double x0[3] = {0.0, 0.0, log(0.02)};
    srukf_set_state(ukf, x0);
    
    // Initial uncertainty
    double S0[9] = {
        0.1, 0.0, 0.0,
        0.0, 0.01, 0.0,
        0.0, 0.0, 0.1
    };
    srukf_set_sqrt_cov(ukf, S0);
    
    // Dynamics (column-major)
    double F[9] = {
        1.0, 0.0, 0.0,     // Column 0
        1.0, 0.95, 0.0,    // Column 1: trend += velocity, velocity *= 0.95
        0.0, 0.0, 0.98     // Column 2: log_vol *= 0.98
    };
    srukf_set_dynamics(ukf, F);
    
    // Measurement: observe trend only
    double H[3] = {1.0, 0.0, 0.0};
    srukf_set_measurement(ukf, H);
    
    // Noise
    double Sq[9] = {0.01, 0, 0, 0, 0.001, 0, 0, 0, 0.01};
    double R0[1] = {1.0};  // Base noise, scaled by exp(2*xi)
    srukf_set_process_noise(ukf, Sq);
    srukf_set_measurement_noise(ukf, R0);
    
    // Run filter
    for (int t = 0; t < 1000; t++) {
        double z[1] = {get_observation(t)};
        srukf_step(ukf, z);
        
        const double* x = srukf_get_state(ukf);
        double w = srukf_get_student_weight(ukf);
        
        printf("t=%d: trend=%.3f, vol=%.4f, weight=%.2f\n",
               t, x[0], srukf_get_volatility(ukf), w);
    }
    
    srukf_destroy(ukf);
    return 0;
}
```

### Integration with Kill Switch

```c
typedef struct {
    StudentT_SRUKF* ukf;
    double position;
    bool halted;
} TradingState;

void trading_loop(TradingState* state) {
    // Enable NIS tracking
    srukf_enable_nis_tracking(state->ukf, 50, 3.0);
    
    while (market_open()) {
        double z[1] = {get_price()};
        
        // Filter step
        srukf_step(state->ukf, z);
        
        // Kill switch checks
        if (!srukf_check_state_bounds(state->ukf, 1e6)) {
            halt_trading(state, "State explosion");
            continue;
        }
        
        if (!srukf_check_covariance(state->ukf)) {
            if (srukf_repair_covariance(state->ukf, 1e-8)) {
                log_warning("Covariance repaired");
            }
        }
        
        if (!srukf_nis_healthy(state->ukf)) {
            reduce_position(state, 0.5);  // Cut position in half
            log_warning("NIS degraded - reducing exposure");
        }
        
        // Normal trading logic
        if (!state->halted) {
            const double* x = srukf_get_state(state->ukf);
            double kelly = compute_kelly(x, srukf_get_volatility(state->ukf));
            update_position(state, kelly);
        }
    }
}
```

### Handling Market Closures

```c
void end_of_day(StudentT_SRUKF* ukf) {
    // Save state
    size_t size = srukf_serialize_size(ukf);
    void* buf = malloc(size);
    srukf_serialize(ukf, buf, size);
    save_to_disk("ukf_state.bin", buf, size);
    free(buf);
}

void start_of_day(StudentT_SRUKF* ukf) {
    // Restore state
    void* buf = load_from_disk("ukf_state.bin");
    srukf_deserialize(ukf, buf, size);
    free(buf);
    
    // Account for overnight gap: run predict steps
    int hours_closed = 14;  // e.g., 4pm to 6am
    for (int h = 0; h < hours_closed; h++) {
        srukf_predict_only(ukf);  // Uncertainty grows
    }
}
```

---

## Performance

### Benchmarks (Intel Core i7, MKL)

```
Student-t SQR UKF Benchmark
============================
Predict step:
  Predict (nx=3): 830 ns/call
  Predict (nx=5): 997 ns/call
  Predict (nx=10): 1670 ns/call
Update step:
  Update (nx=3, nz=1): 1451 ns/call
  Update (nx=5, nz=1): 1693 ns/call
  Update (nx=10, nz=3): 3004 ns/call
Full step (predict + update):
  Step (nx=3, nz=1): 2369 ns/call | 0.42M steps/sec
  Step (nx=5, nz=1): 2735 ns/call | 0.37M steps/sec
  Step (nx=10, nz=3): 4723 ns/call | 0.21M steps/sec
```

### Comparison with Stack

| Component | Latency | Notes |
|-----------|---------|-------|
| SSA | 600 µs | Dominates pipeline |
| BOCPD | ~100 ns | Negligible |
| **UKF** | **2.4 µs** | Fast |
| Kelly | ~10 ns | Trivial |

Total pipeline: ~603 µs, SSA-dominated.

---

## Integration

### MKL Configuration

For maximum performance and reproducibility, MKL must be configured correctly.

#### Configuration Files Provided

| File | Purpose |
|------|---------|
| `mkl_config.h` | C header with programmatic configuration |
| `run_mkl_optimized.bat` | Windows startup script |
| `run_mkl_optimized.sh` | Linux startup script |

#### Key Settings for Quant Trading

| Setting | Value | Rationale |
|---------|-------|-----------|
| `MKL_NUM_THREADS` | 1 | Single-threaded for minimum latency (nx < 20) |
| `MKL_DYNAMIC` | FALSE | Fixed thread count, no runtime overhead |
| `MKL_ENABLE_INSTRUCTIONS` | AVX512 | Maximum SIMD width |
| `MKL_CBWR` | AUTO | Deterministic results for reproducibility |
| `MKL_JIT_MAX_SIZE` | 64 | JIT code gen for small matrices |

#### Programmatic Configuration

```c
#include "mkl_config.h"

int main() {
    // Option 1: Default quant trading config
    mkl_config_init();
    
    // Option 2: Specific preset
    mkl_config_init_preset(MKL_CONFIG_QUANT_LATENCY);
    
    // Option 3: Custom configuration
    MKL_Config cfg = mkl_config_default();
    cfg.num_threads = 2;  // Use 2 threads
    cfg.deterministic = true;
    cfg.verbose = true;
    mkl_config_apply(&cfg);
    
    // Print configuration for logging
    mkl_config_print_info();
    
    // Verify configuration
    mkl_config_verify();
    
    // ... your code ...
}
```

#### Script-Based Configuration

**Windows:**
```batch
run_mkl_optimized.bat build\Release\my_trading_bot.exe
```

**Linux:**
```bash
chmod +x run_mkl_optimized.sh
./run_mkl_optimized.sh ./build/my_trading_bot
```

#### Why These Settings?

**Single-threaded (`MKL_NUM_THREADS=1`):**
- For matrices smaller than ~100×100, threading overhead exceeds benefit
- UKF with nx=3 uses 3×7 matrices — threading adds ~500ns latency
- Single-threaded: predictable, consistent latency

**AVX512 (`MKL_ENABLE_INSTRUCTIONS=AVX512`):**
- 512-bit vectors process 8 doubles per instruction
- Even with frequency throttling, faster for small matrices
- Automatic fallback if not supported

**Determinism (`MKL_CBWR=AUTO`):**
- Same inputs → same outputs (critical for debugging, backtesting)
- Slight overhead (~5%) but essential for production
- `AUTO` mode picks best SIMD while maintaining reproducibility

**JIT (`MKL_JIT_MAX_SIZE=64`):**
- MKL generates optimized code for specific matrix sizes
- First call: ~10µs overhead (code generation)
- Subsequent calls: 2-3× faster than generic BLAS
- UKF calls same sizes repeatedly — perfect for JIT

#### Expected Output

```
╔══════════════════════════════════════════════════════════════════╗
║                    MKL CONFIGURATION                             ║
╠══════════════════════════════════════════════════════════════════╣
║ MKL Version: 2024.0.0 (Product)
║ Build: 20231006
╠══════════════════════════════════════════════════════════════════╣
║ Threading:                                                       ║
║   Max threads: 1
║   Dynamic: disabled
╠══════════════════════════════════════════════════════════════════╣
║ Determinism:                                                     ║
║   CNR Mode: AUTO
╠══════════════════════════════════════════════════════════════════╣
║ CPU Features:                                                    ║
║   AVX2:   available
║   AVX512: available
╠══════════════════════════════════════════════════════════════════╣
║ Memory:                                                          ║
║   Recommended alignment: 64 bytes (AVX512)                       ║
╚══════════════════════════════════════════════════════════════════╝
```

### Build Requirements

- CMake 3.16+
- Intel MKL (oneAPI)
- C11 compiler (MSVC, GCC, Clang)

### CMake

```cmake
find_package(MKL CONFIG REQUIRED)
target_link_libraries(your_target PRIVATE student_t_srukf MKL::MKL)
```

### Build Commands

```bash
# Windows (MSVC)
mkdir build && cd build
cmake ..
cmake --build . --config Release

# Linux
mkdir build && cd build
source /opt/intel/oneapi/setvars.sh
cmake ..
make
```

### File Structure

```
SQR-UKF-MKL/
├── MKL/
│   ├── student_t_srukf.c    # Implementation
│   ├── student_t_srukf.h    # Header
│   └── mkl_config.h         # MKL configuration
├── test/
│   ├── test_srukf.c         # Unit tests
│   └── bench_srukf.c        # Benchmarks
├── CMakeLists.txt
├── run_mkl_optimized.bat    # Windows startup script
├── run_mkl_optimized.sh     # Linux startup script
├── STUDENT_T_SRUKF.md       # This document
├── LICENSE
└── README.md
```

---

## References

1. Julier, S.J. & Uhlmann, J.K. (1997). "A New Extension of the Kalman Filter to Nonlinear Systems"
2. Van der Merwe, R. & Wan, E.A. (2001). "The Square-Root Unscented Kalman Filter for State and Parameter-Estimation"
3. Agamennoni, G. et al. (2012). "An Outlier-Robust Kalman Filter" (Student-t modification)
4. Intel Math Kernel Library Documentation

---

## Version History

| Version | Changes |
|---------|---------|
| 1.0 | Initial release: core UKF, Student-t, MKL optimization |
| 1.1 | Added: missing data, covariance repair, NIS tracking, serialization |

---

## License

See LICENSE file.
