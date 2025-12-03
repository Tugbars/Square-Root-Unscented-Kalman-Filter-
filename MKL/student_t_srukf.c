/**
 * @file student_t_srukf.c
 * @brief Student-t Square-Root UKF Implementation with Intel MKL
 *
 * Optimizations:
 * - BLAS 3 (dgemm, dsyrk) wherever possible
 * - Vectorized exp via vdExp (MKL VML)
 * - Single contiguous memory allocation
 * - 64-byte alignment for AVX512
 * - restrict pointers for aliasing hints
 * - Zero allocations in hot path
 */

#include "student_t_srukf.h"

#include <mkl.h>
#include <mkl_vml.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

/*─────────────────────────────────────────────────────────────────────────────
 * CONSTANTS
 *───────────────────────────────────────────────────────────────────────────*/

#define ALIGN 64 /* Cache line / AVX512 alignment */

/* Macro for aligned offset calculation */
#define ALIGN_UP(size) (((size) + ALIGN - 1) / ALIGN * ALIGN)

/* Forward declaration */
static inline void update_nis_tracking(StudentT_SRUKF *ukf, double nis);

/*─────────────────────────────────────────────────────────────────────────────
 * INTERNAL STRUCTURE
 *───────────────────────────────────────────────────────────────────────────*/

struct StudentT_SRUKF
{
    /* Dimensions */
    int nx;    /* State dimension */
    int nz;    /* Measurement dimension */
    int n_sig; /* Number of sigma points: 2*nx + 1 */

    /* Volatility state index */
    int xi_index; /* Which state is log(sigma) */

    /* UKF parameters */
    double alpha;  /* Sigma point spread */
    double beta;   /* Prior distribution (2.0 = Gaussian) */
    double kappa;  /* Secondary scaling */
    double lambda; /* Composite: alpha²(nx + kappa) - nx */
    double gamma;  /* sqrt(nx + lambda) */

    /* Student-t */
    double nu; /* Degrees of freedom */

    /* State and covariance (column-major) */
    double *restrict x; /* [nx] current state */
    double *restrict S; /* [nx × nx] sqrt covariance, lower triangular */

    /* Model matrices */
    double *restrict F;  /* [nx × nx] dynamics */
    double *restrict H;  /* [nz × nx] measurement */
    double *restrict Sq; /* [nx × nx] process noise sqrt */
    double *restrict R0; /* [nz × nz] base measurement noise sqrt */

    /* Weights */
    double *restrict Wm;      /* [n_sig] mean weights */
    double *restrict Wc;      /* [n_sig] covariance weights */
    double *restrict Wc_sqrt; /* [n_sig] precomputed sqrt(|Wc|) * sign(Wc) */

    /* Workspace - sigma points */
    double *restrict X_sig;  /* [nx × n_sig] sigma points */
    double *restrict X_pred; /* [nx × n_sig] propagated sigma points */
    double *restrict Z_sig;  /* [nz × n_sig] measurement sigma points */

    /* Workspace - means */
    double *restrict x_pred;     /* [nx] predicted state mean */
    double *restrict z_pred;     /* [nz] predicted measurement mean */
    double *restrict innovation; /* [nz] z - z_pred */

    /* Workspace - covariances */
    double *restrict Pxz; /* [nx × nz] cross covariance */
    double *restrict K;   /* [nx × nz] Kalman gain */
    double *restrict Szz; /* [nz × nz] innovation sqrt covariance */
    double *restrict U;   /* [nx × nz] K @ Szz for downdate */

    /* Workspace - state-dependent noise */
    double *restrict xi_work;  /* [n_sig] temp for 2*xi values */
    double *restrict R_scales; /* [n_sig] exp(2*xi) per sigma point */

    /* Workspace - Cholesky */
    double *restrict work_mat; /* General workspace matrix */

    /* Workspace - downdate */
    double *restrict c_work; /* [nx] Givens c values */
    double *restrict s_work; /* [nx] Givens s values */

    /* Diagnostics */
    double nis;            /* Normalized innovation squared */
    double mahalanobis_sq; /* Mahalanobis distance squared */
    double student_weight; /* Last Student-t weight */

    /* NIS windowed statistics */
    bool nis_tracking_enabled;
    int nis_window_size;
    double nis_outlier_threshold;
    int nis_window_idx;           /* Circular buffer index */
    int nis_window_fill;          /* How many samples collected */
    double *restrict nis_history; /* [window_size] circular buffer */
    double nis_sum;               /* Running sum for mean */
    double nis_sum_sq;            /* Running sum of squares for variance */

    /* Memory management */
    void *_memory; /* Single aligned allocation */
    size_t _memory_size;
};

/*─────────────────────────────────────────────────────────────────────────────
 * HELPER: CHOLESKY RANK-1 DOWNDATE
 *
 * Computes S_new such that S_new @ S_new' = S @ S' - u @ u'
 * S is lower triangular, modified in-place
 *───────────────────────────────────────────────────────────────────────────*/

static inline void cholesky_downdate(
    double *restrict S,       /* [n × n] lower triangular, modified in-place */
    const double *restrict u, /* [n] vector to downdate with */
    double *restrict u_work,  /* [n] workspace (modified) */
    double *restrict c,       /* [n] workspace for c values */
    double *restrict s,       /* [n] workspace for s values */
    int n)
{
    /* Copy u to workspace */
    cblas_dcopy(n, u, 1, u_work, 1);

    for (int k = 0; k < n; k++)
    {
        double S_kk = S[k + k * n];
        double u_k = u_work[k];

        /* Check for numerical issues */
        double r_sq = S_kk * S_kk - u_k * u_k;
        if (r_sq <= 0.0)
        {
            /* Matrix would become non-positive-definite */
            /* Fall back to small positive value */
            r_sq = 1e-12 * S_kk * S_kk;
        }

        double r = sqrt(r_sq);
        c[k] = r / S_kk;
        s[k] = u_k / S_kk;
        S[k + k * n] = r;

        if (k < n - 1)
        {
            /* Update remaining rows */
            for (int j = k + 1; j < n; j++)
            {
                double S_jk = S[j + k * n];
                S[j + k * n] = (S_jk - s[k] * u_work[j]) / c[k];
                u_work[j] = c[k] * u_work[j] - s[k] * S_jk;
            }
        }
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * CREATION / DESTRUCTION
 *───────────────────────────────────────────────────────────────────────────*/

StudentT_SRUKF *srukf_create(int nx, int nz, double nu)
{
    /* Validate inputs */
    if (nx <= 0 || nz <= 0 || nu <= 2.0)
    {
        return NULL; /* nu must be > 2 for finite variance */
    }

    /* Allocate struct (aligned) */
    StudentT_SRUKF *ukf = (StudentT_SRUKF *)mkl_malloc(sizeof(StudentT_SRUKF), ALIGN);
    if (!ukf)
        return NULL;

    memset(ukf, 0, sizeof(StudentT_SRUKF));

    /* Store dimensions */
    ukf->nx = nx;
    ukf->nz = nz;
    ukf->n_sig = 2 * nx + 1;
    ukf->nu = nu;
    ukf->xi_index = nx - 1; /* Default: last state is log-vol */

    /* Default UKF parameters */
    ukf->alpha = 1e-3;
    ukf->beta = 2.0;
    ukf->kappa = 0.0;

    int n_sig = ukf->n_sig;

    /* Calculate memory layout with alignment */
    size_t offset = 0;

    /* State and covariance */
    size_t off_x = offset;
    offset += ALIGN_UP(nx * sizeof(double));
    size_t off_S = offset;
    offset += ALIGN_UP(nx * nx * sizeof(double));

    /* Model matrices */
    size_t off_F = offset;
    offset += ALIGN_UP(nx * nx * sizeof(double));
    size_t off_H = offset;
    offset += ALIGN_UP(nz * nx * sizeof(double));
    size_t off_Sq = offset;
    offset += ALIGN_UP(nx * nx * sizeof(double));
    size_t off_R0 = offset;
    offset += ALIGN_UP(nz * nz * sizeof(double));

    /* Weights */
    size_t off_Wm = offset;
    offset += ALIGN_UP(n_sig * sizeof(double));
    size_t off_Wc = offset;
    offset += ALIGN_UP(n_sig * sizeof(double));
    size_t off_Wc_sqrt = offset;
    offset += ALIGN_UP(n_sig * sizeof(double));

    /* Sigma points */
    size_t off_X_sig = offset;
    offset += ALIGN_UP(nx * n_sig * sizeof(double));
    size_t off_X_pred = offset;
    offset += ALIGN_UP(nx * n_sig * sizeof(double));
    size_t off_Z_sig = offset;
    offset += ALIGN_UP(nz * n_sig * sizeof(double));

    /* Means */
    size_t off_x_pred = offset;
    offset += ALIGN_UP(nx * sizeof(double));
    size_t off_z_pred = offset;
    offset += ALIGN_UP(nz * sizeof(double));
    size_t off_innov = offset;
    offset += ALIGN_UP(nz * sizeof(double));

    /* Covariances and gain */
    size_t off_Pxz = offset;
    offset += ALIGN_UP(nx * nz * sizeof(double));
    size_t off_K = offset;
    offset += ALIGN_UP(nx * nz * sizeof(double));
    size_t off_Szz = offset;
    offset += ALIGN_UP(nz * nz * sizeof(double));
    size_t off_U = offset;
    offset += ALIGN_UP(nx * nz * sizeof(double));

    /* State-dependent noise */
    size_t off_xi_work = offset;
    offset += ALIGN_UP(n_sig * sizeof(double));
    size_t off_R_scales = offset;
    offset += ALIGN_UP(n_sig * sizeof(double));

    /* General workspace: Z_c [nz×n_sig] + Z_centered [nz×n_sig] + X_c [nx×n_sig] */
    size_t work_mat_size = (2 * nz + nx) * n_sig;
    size_t off_work_mat = offset;
    offset += ALIGN_UP(work_mat_size * sizeof(double));

    /* Downdate workspace */
    size_t off_c_work = offset;
    offset += ALIGN_UP(nx * sizeof(double));
    size_t off_s_work = offset;
    offset += ALIGN_UP(nx * sizeof(double));

    /* Allocate single memory block */
    ukf->_memory_size = offset;
    ukf->_memory = mkl_malloc(offset, ALIGN);
    if (!ukf->_memory)
    {
        mkl_free(ukf);
        return NULL;
    }
    memset(ukf->_memory, 0, offset);

    /* Assign pointers */
    char *base = (char *)ukf->_memory;

    ukf->x = (double *)(base + off_x);
    ukf->S = (double *)(base + off_S);
    ukf->F = (double *)(base + off_F);
    ukf->H = (double *)(base + off_H);
    ukf->Sq = (double *)(base + off_Sq);
    ukf->R0 = (double *)(base + off_R0);
    ukf->Wm = (double *)(base + off_Wm);
    ukf->Wc = (double *)(base + off_Wc);
    ukf->Wc_sqrt = (double *)(base + off_Wc_sqrt);
    ukf->X_sig = (double *)(base + off_X_sig);
    ukf->X_pred = (double *)(base + off_X_pred);
    ukf->Z_sig = (double *)(base + off_Z_sig);
    ukf->x_pred = (double *)(base + off_x_pred);
    ukf->z_pred = (double *)(base + off_z_pred);
    ukf->innovation = (double *)(base + off_innov);
    ukf->Pxz = (double *)(base + off_Pxz);
    ukf->K = (double *)(base + off_K);
    ukf->Szz = (double *)(base + off_Szz);
    ukf->U = (double *)(base + off_U);
    ukf->xi_work = (double *)(base + off_xi_work);
    ukf->R_scales = (double *)(base + off_R_scales);
    ukf->work_mat = (double *)(base + off_work_mat);
    ukf->c_work = (double *)(base + off_c_work);
    ukf->s_work = (double *)(base + off_s_work);

    /* Initialize weights */
    srukf_set_ukf_params(ukf, ukf->alpha, ukf->beta, ukf->kappa);

    /* Initialize diagnostics */
    ukf->nis = 0.0;
    ukf->mahalanobis_sq = 0.0;
    ukf->student_weight = 1.0;

    /* NIS tracking disabled by default (enable with srukf_enable_nis_tracking) */
    ukf->nis_tracking_enabled = false;
    ukf->nis_window_size = 0;
    ukf->nis_outlier_threshold = 0.0;
    ukf->nis_window_idx = 0;
    ukf->nis_window_fill = 0;
    ukf->nis_history = NULL;
    ukf->nis_sum = 0.0;
    ukf->nis_sum_sq = 0.0;

    return ukf;
}

void srukf_destroy(StudentT_SRUKF *ukf)
{
    if (ukf)
    {
        if (ukf->nis_history)
            mkl_free(ukf->nis_history);
        if (ukf->_memory)
            mkl_free(ukf->_memory);
        mkl_free(ukf);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_set_ukf_params(StudentT_SRUKF *ukf, double alpha, double beta, double kappa)
{
    ukf->alpha = alpha;
    ukf->beta = beta;
    ukf->kappa = kappa;

    int nx = ukf->nx;
    int n_sig = ukf->n_sig;

    /* Compute composite parameters */
    ukf->lambda = alpha * alpha * (nx + kappa) - nx;
    ukf->gamma = sqrt(nx + ukf->lambda);

    double lambda = ukf->lambda;

    /* Mean weights */
    ukf->Wm[0] = lambda / (nx + lambda);
    for (int i = 1; i < n_sig; i++)
    {
        ukf->Wm[i] = 0.5 / (nx + lambda);
    }

    /* Covariance weights */
    ukf->Wc[0] = lambda / (nx + lambda) + (1.0 - alpha * alpha + beta);
    for (int i = 1; i < n_sig; i++)
    {
        ukf->Wc[i] = 0.5 / (nx + lambda);
    }

    /* Precompute sqrt(|Wc|) * sign(Wc) for hot path */
    for (int i = 0; i < n_sig; i++)
    {
        double sw = sqrt(fabs(ukf->Wc[i]));
        if (ukf->Wc[i] < 0.0)
            sw = -sw;
        ukf->Wc_sqrt[i] = sw;
    }
}

void srukf_set_state(StudentT_SRUKF *restrict ukf, const double *restrict x0)
{
    cblas_dcopy(ukf->nx, x0, 1, ukf->x, 1);
}

void srukf_set_sqrt_cov(StudentT_SRUKF *restrict ukf, const double *restrict S0)
{
    cblas_dcopy(ukf->nx * ukf->nx, S0, 1, ukf->S, 1);
}

void srukf_set_dynamics(StudentT_SRUKF *restrict ukf, const double *restrict F)
{
    cblas_dcopy(ukf->nx * ukf->nx, F, 1, ukf->F, 1);
}

void srukf_set_measurement(StudentT_SRUKF *restrict ukf, const double *restrict H)
{
    cblas_dcopy(ukf->nz * ukf->nx, H, 1, ukf->H, 1);
}

void srukf_set_process_noise(StudentT_SRUKF *restrict ukf, const double *restrict Sq)
{
    cblas_dcopy(ukf->nx * ukf->nx, Sq, 1, ukf->Sq, 1);
}

void srukf_set_measurement_noise(StudentT_SRUKF *restrict ukf, const double *restrict R0)
{
    cblas_dcopy(ukf->nz * ukf->nz, R0, 1, ukf->R0, 1);
}

void srukf_set_vol_index(StudentT_SRUKF *ukf, int xi_index)
{
    ukf->xi_index = xi_index;
}

void srukf_set_student_nu(StudentT_SRUKF *ukf, double nu)
{
    ukf->nu = nu;
}

/*─────────────────────────────────────────────────────────────────────────────
 * SIGMA POINT GENERATION
 *───────────────────────────────────────────────────────────────────────────*/

static inline void generate_sigma_points(
    const double *restrict x, /* [nx] state */
    const double *restrict S, /* [nx × nx] sqrt covariance, lower triangular */
    double gamma,             /* sqrt(nx + lambda) */
    double *restrict X_sig,   /* [nx × n_sig] output sigma points */
    int nx,
    int n_sig)
{
    /* Column 0: X[:,0] = x */
    cblas_dcopy(nx, x, 1, X_sig, 1);

    /* Columns 1 to nx: X[:,i] = x + gamma * S[:,i-1] */
    for (int i = 1; i <= nx; i++)
    {
        cblas_dcopy(nx, x, 1, X_sig + i * nx, 1);
    }
    cblas_daxpy(nx * nx, gamma, S, 1, X_sig + nx, 1);

    /* Columns nx+1 to 2nx: X[:,i] = x - gamma * S[:,i-nx-1] */
    for (int i = nx + 1; i < n_sig; i++)
    {
        cblas_dcopy(nx, x, 1, X_sig + i * nx, 1);
    }
    cblas_daxpy(nx * nx, -gamma, S, 1, X_sig + (nx + 1) * nx, 1);
}

/*─────────────────────────────────────────────────────────────────────────────
 * PREDICT STEP
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_predict(StudentT_SRUKF *restrict ukf)
{
    const int nx = ukf->nx;
    const int n_sig = ukf->n_sig;

    /*───────────────────────────────────────────────────────────────────────
     * 1. Generate sigma points
     *─────────────────────────────────────────────────────────────────────*/
    generate_sigma_points(ukf->x, ukf->S, ukf->gamma, ukf->X_sig, nx, n_sig);

    /*───────────────────────────────────────────────────────────────────────
     * 2. Propagate through dynamics: X_pred = F @ X_sig  (BLAS 3)
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                nx, n_sig, nx,
                1.0, ukf->F, nx,
                ukf->X_sig, nx,
                0.0, ukf->X_pred, nx);

    /*───────────────────────────────────────────────────────────────────────
     * 3. Compute predicted mean: x_pred = X_pred @ Wm  (BLAS 2)
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nx, n_sig,
                1.0, ukf->X_pred, nx,
                ukf->Wm, 1,
                0.0, ukf->x_pred, 1);

    /*───────────────────────────────────────────────────────────────────────
     * 4. Compute predicted sqrt covariance
     *    Using dsyrk for P = X_centered @ diag(Wc) @ X_centered' + Q
     *    Then Cholesky to get sqrt form
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict X_c = ukf->work_mat;
    const double *restrict X_pred = ukf->X_pred;
    const double *restrict x_pred = ukf->x_pred;
    const double *restrict Wc_sqrt = ukf->Wc_sqrt;

    /* Build centered and weighted sigma points (Wc_sqrt precomputed with sign) */
    for (int i = 0; i < n_sig; i++)
    {
        const double sw = Wc_sqrt[i];
        for (int j = 0; j < nx; j++)
        {
            X_c[j + i * nx] = sw * (X_pred[j + i * nx] - x_pred[j]);
        }
    }

    /* P = X_c @ X_c' via dsyrk (BLAS 3, exploits symmetry) */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                nx, n_sig,
                1.0, X_c, nx,
                0.0, ukf->S, nx);

    /* Add Q = Sq @ Sq' */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                nx, nx,
                1.0, ukf->Sq, nx,
                1.0, ukf->S, nx);

    /* Cholesky factorization: S = chol(P, 'lower') */
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nx, ukf->S, nx);

    /*───────────────────────────────────────────────────────────────────────
     * 5. Copy predicted mean to state
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dcopy(nx, ukf->x_pred, 1, ukf->x, 1);
}

/*─────────────────────────────────────────────────────────────────────────────
 * UPDATE STEP - SCALAR MEASUREMENT (nz=1) FAST PATH
 *
 * When nz=1, many matrix operations collapse to scalars/vectors:
 * - dsyrk → ddot (scalar variance)
 * - dtrsm × 2 → scalar division
 * - dgemm for Pxz → dgemv
 * - Single rank-1 downdate instead of loop
 *
 * ~50% faster than general case for nz=1
 *───────────────────────────────────────────────────────────────────────────*/

static void srukf_update_scalar(StudentT_SRUKF *restrict ukf, double z)
{
    const int nx = ukf->nx;
    const int n_sig = ukf->n_sig;
    const int xi_idx = ukf->xi_index;
    const double nu = ukf->nu;

    const double *restrict X_pred = ukf->X_pred;
    const double *restrict H = ukf->H; /* 1 × nx row vector */
    const double *restrict Wm = ukf->Wm;
    const double *restrict Wc = ukf->Wc;

    /*───────────────────────────────────────────────────────────────────────
     * 1. Compute R_scales = exp(2 * xi) for each sigma point
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict xi_work = ukf->xi_work;
    double *restrict R_scales = ukf->R_scales;

    for (int i = 0; i < n_sig; i++)
    {
        xi_work[i] = 2.0 * X_pred[xi_idx + i * nx];
    }
    vdExp(n_sig, xi_work, R_scales);

    /*───────────────────────────────────────────────────────────────────────
     * 2. Measurement sigma points: z_sig[i] = H @ X_pred[:,i]
     *    For nz=1, Z_sig is just a vector of length n_sig
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict z_sig = ukf->Z_sig; /* Reuse as n_sig vector */

    for (int i = 0; i < n_sig; i++)
    {
        z_sig[i] = cblas_ddot(nx, H, 1, X_pred + i * nx, 1);
    }

    /*───────────────────────────────────────────────────────────────────────
     * 3. Predicted measurement: z_pred = sum(Wm[i] * z_sig[i])
     *─────────────────────────────────────────────────────────────────────*/
    double z_pred = cblas_ddot(n_sig, Wm, 1, z_sig, 1);
    ukf->z_pred[0] = z_pred;

    /*───────────────────────────────────────────────────────────────────────
     * 4. Innovation
     *─────────────────────────────────────────────────────────────────────*/
    double innov = z - z_pred;
    ukf->innovation[0] = innov;

    /*───────────────────────────────────────────────────────────────────────
     * 5. Innovation variance: Pzz = sum(Wc[i] * (z_sig[i] - z_pred)²) + R
     *    Szz = sqrt(Pzz) (scalar)
     *─────────────────────────────────────────────────────────────────────*/
    double Pzz = 0.0;
    double avg_R_scale = 0.0;

    for (int i = 0; i < n_sig; i++)
    {
        double dz = z_sig[i] - z_pred;
        Pzz += Wc[i] * dz * dz;
        avg_R_scale += Wm[i] * R_scales[i];
    }

    /* Add measurement noise: R = R0² * avg_R_scale */
    double R0_sq = ukf->R0[0] * ukf->R0[0];
    Pzz += avg_R_scale * R0_sq;

    double Szz = sqrt(Pzz);
    ukf->Szz[0] = Szz;

    /*───────────────────────────────────────────────────────────────────────
     * 6. Cross-covariance: Pxz = sum(Wc[i] * (X_pred[:,i] - x) * (z_sig[i] - z_pred))
     *    This is an nx-vector for nz=1
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict Pxz = ukf->Pxz; /* nx × 1 */
    const double *restrict x = ukf->x;

    /* Initialize to zero */
    memset(Pxz, 0, nx * sizeof(double));

    /* Accumulate weighted outer products */
    for (int i = 0; i < n_sig; i++)
    {
        double w_dz = Wc[i] * (z_sig[i] - z_pred);
        cblas_daxpy(nx, w_dz, X_pred + i * nx, 1, Pxz, 1);
    }

    /* Subtract mean contribution: Pxz -= x * sum(Wc) * 0 (z_pred - z_pred = 0) */
    /* Actually: Pxz = sum(Wc[i] * X_pred[:,i] * dz[i]) - x * sum(Wc[i] * dz[i]) */
    /* The second term is zero because sum(Wc[i] * dz[i]) = z_pred - z_pred = 0 */
    /* But we need to handle it properly: */
    double sum_Wc_dz = 0.0;
    for (int i = 0; i < n_sig; i++)
    {
        sum_Wc_dz += Wc[i] * (z_sig[i] - z_pred);
    }
    cblas_daxpy(nx, -sum_Wc_dz, x, 1, Pxz, 1);

    /*───────────────────────────────────────────────────────────────────────
     * 7. Kalman gain: K = Pxz / Pzz  (scalar division for nz=1)
     *    K is nx × 1 vector
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict K = ukf->K;
    double inv_Pzz = 1.0 / Pzz;

    for (int i = 0; i < nx; i++)
    {
        K[i] = Pxz[i] * inv_Pzz;
    }

    /*───────────────────────────────────────────────────────────────────────
     * 8. Mahalanobis distance: d² = innov² / Pzz
     *─────────────────────────────────────────────────────────────────────*/
    double d_sq = innov * innov * inv_Pzz;
    ukf->mahalanobis_sq = d_sq;

    /*───────────────────────────────────────────────────────────────────────
     * 9. Student-t weight
     *─────────────────────────────────────────────────────────────────────*/
    double w = (nu + 1.0) / (nu + d_sq); /* nz=1 */
    if (w > 1.0)
        w = 1.0;
    ukf->student_weight = w;

    /*───────────────────────────────────────────────────────────────────────
     * 10. NIS for diagnostics
     *─────────────────────────────────────────────────────────────────────*/
    ukf->nis = d_sq;
    update_nis_tracking(ukf, d_sq);

    /*───────────────────────────────────────────────────────────────────────
     * 11. State update: x = x + w * K * innov
     *─────────────────────────────────────────────────────────────────────*/
    cblas_daxpy(nx, w * innov, K, 1, ukf->x, 1);

    /*───────────────────────────────────────────────────────────────────────
     * 12. Covariance downdate: single rank-1 update
     *     U = sqrt(w) * K * Szz  →  just sqrt(w) * Szz * K (scale K)
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict u_vec = ukf->U; /* Reuse as nx vector */
    double scale = sqrt(w) * Szz;

    for (int i = 0; i < nx; i++)
    {
        u_vec[i] = scale * K[i];
    }

    /* Single rank-1 downdate */
    cholesky_downdate(
        ukf->S,
        u_vec,
        ukf->work_mat, /* u_work */
        ukf->c_work,
        ukf->s_work,
        nx);
}

/*─────────────────────────────────────────────────────────────────────────────
 * UPDATE STEP (with Student-t weighting)
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_update(StudentT_SRUKF *restrict ukf, const double *restrict z)
{
    /* Fast path for scalar measurement (very common case) */
    if (ukf->nz == 1)
    {
        srukf_update_scalar(ukf, z[0]);
        return;
    }

    const int nx = ukf->nx;
    const int nz = ukf->nz;
    const int n_sig = ukf->n_sig;
    const int xi_idx = ukf->xi_index;
    const double nu = ukf->nu;

    /*───────────────────────────────────────────────────────────────────────
     * 1. Reuse predicted sigma points (X_pred from predict step)
     *    No regeneration needed - saves O(nx²) and avoids roundoff
     *─────────────────────────────────────────────────────────────────────*/
    const double *restrict X_pred = ukf->X_pred;

    /*───────────────────────────────────────────────────────────────────────
     * 2. Compute state-dependent measurement noise scales
     *    R_scales[i] = exp(2 * xi[i])  using vectorized exp
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict xi_work = ukf->xi_work;
    double *restrict R_scales = ukf->R_scales;

    /* Extract xi values from X_pred and multiply by 2 */
    for (int i = 0; i < n_sig; i++)
    {
        xi_work[i] = 2.0 * X_pred[xi_idx + i * nx];
    }

    /* Vectorized exp: R_scales = exp(xi_work) */
    vdExp(n_sig, xi_work, R_scales);

    /*───────────────────────────────────────────────────────────────────────
     * 3. Measurement sigma points: Z_sig = H @ X_pred  (BLAS 3)
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                nz, n_sig, nx,
                1.0, ukf->H, nz,
                X_pred, nx,
                0.0, ukf->Z_sig, nz);

    /*───────────────────────────────────────────────────────────────────────
     * 4. Predicted measurement: z_pred = Z_sig @ Wm  (BLAS 2)
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nz, n_sig,
                1.0, ukf->Z_sig, nz,
                ukf->Wm, 1,
                0.0, ukf->z_pred, 1);

    /*───────────────────────────────────────────────────────────────────────
     * 5. Innovation
     *─────────────────────────────────────────────────────────────────────*/
    for (int i = 0; i < nz; i++)
    {
        ukf->innovation[i] = z[i] - ukf->z_pred[i];
    }

    /*───────────────────────────────────────────────────────────────────────
     * 6. Innovation covariance Szz via dsyrk
     *    Also build Z_centered for cross-covariance (fused loop)
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict Z_c = ukf->work_mat;
    double *restrict Z_centered = ukf->work_mat + nz * n_sig;
    const double *restrict Z_sig = ukf->Z_sig;
    const double *restrict z_pred = ukf->z_pred;
    const double *restrict Wc_sqrt = ukf->Wc_sqrt;

    /* Fused loop: compute Z_centered and Z_c in one pass (Fix 3) */
    for (int i = 0; i < n_sig; i++)
    {
        const double sw = Wc_sqrt[i]; /* Precomputed with sign (Fix 2+5) */
        for (int j = 0; j < nz; j++)
        {
            const double diff = Z_sig[j + i * nz] - z_pred[j];
            Z_centered[j + i * nz] = diff;
            Z_c[j + i * nz] = sw * diff;
        }
    }

    /* Szz = Z_c @ Z_c' */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                nz, n_sig,
                1.0, Z_c, nz,
                0.0, ukf->Szz, nz);

    /* Add weighted average of R0 scaled by R_scales */
    double avg_R_scale = cblas_ddot(n_sig, ukf->Wm, 1, R_scales, 1);

    /* Szz += avg_R_scale * (R0 @ R0') */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                nz, nz,
                avg_R_scale, ukf->R0, nz,
                1.0, ukf->Szz, nz);

    /* Cholesky of Szz */
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', nz, ukf->Szz, nz);

    /*───────────────────────────────────────────────────────────────────────
     * 7. Cross-covariance Pxz  (BLAS 3)
     *    Pxz = sum_i Wc[i] * (X_pred[:,i] - x) @ (Z_sig[:,i] - z_pred)'
     *    Z_centered already computed in step 6 (fused loop)
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict X_c = ukf->work_mat + 2 * nz * n_sig; /* Offset past Z_c and Z_centered */
    const double *restrict x = ukf->x;
    const double *restrict Wc = ukf->Wc;

    /* Build weighted X_centered using X_pred */
    for (int i = 0; i < n_sig; i++)
    {
        const double w = Wc[i];
        for (int j = 0; j < nx; j++)
        {
            X_c[j + i * nx] = w * (X_pred[j + i * nx] - x[j]);
        }
    }

    /* Pxz = X_c @ Z_centered' (Z_centered computed in step 6) */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                nx, nz, n_sig,
                1.0, X_c, nx,
                Z_centered, nz,
                0.0, ukf->Pxz, nx);

    /*───────────────────────────────────────────────────────────────────────
     * 8. Kalman gain: K = Pxz @ Szz^{-T} @ Szz^{-1}
     *    Using two triangular solves (BLAS 3: dtrsm)
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dcopy(nx * nz, ukf->Pxz, 1, ukf->K, 1);

    /* K = Pxz @ inv(Szz') where Szz is lower triangular */
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                nx, nz,
                1.0, ukf->Szz, nz,
                ukf->K, nx);

    /* K = K @ inv(Szz) */
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                nx, nz,
                1.0, ukf->Szz, nz,
                ukf->K, nx);

    /*───────────────────────────────────────────────────────────────────────
     * 9. Mahalanobis distance: d² = innovation' @ inv(Pzz) @ innovation
     *    = ||Szz^{-1} @ innovation||²
     *─────────────────────────────────────────────────────────────────────*/
    double *restrict v = ukf->z_pred; /* Reuse as temp */
    cblas_dcopy(nz, ukf->innovation, 1, v, 1);

    /* Solve Szz @ v = innovation → v = Szz^{-1} @ innovation */
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                nz, ukf->Szz, nz, v, 1);

    double d_sq = cblas_ddot(nz, v, 1, v, 1);
    ukf->mahalanobis_sq = d_sq;

    /*───────────────────────────────────────────────────────────────────────
     * 10. Student-t weight
     *     Cap at 1.0: don't upweight small innovations
     *─────────────────────────────────────────────────────────────────────*/
    double w = (nu + (double)nz) / (nu + d_sq);
    if (w > 1.0)
        w = 1.0;
    ukf->student_weight = w;

    /*───────────────────────────────────────────────────────────────────────
     * 11. NIS for diagnostics / kill switch
     *─────────────────────────────────────────────────────────────────────*/
    ukf->nis = d_sq; /* Under correct model: E[NIS] ≈ nz */

    /* Update NIS tracking if enabled */
    update_nis_tracking(ukf, d_sq);

    /*───────────────────────────────────────────────────────────────────────
     * 12. State update: x = x + w * K @ innovation
     *─────────────────────────────────────────────────────────────────────*/
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nx, nz,
                w, ukf->K, nx,
                ukf->innovation, 1,
                1.0, ukf->x, 1);

    /*───────────────────────────────────────────────────────────────────────
     * 13. Covariance update: S_new where S_new @ S_new' = S @ S' - w * K @ Pzz @ K'
     *     = S @ S' - w * (K @ Szz) @ (K @ Szz)'
     *
     *     Perform rank-nz downdate
     *─────────────────────────────────────────────────────────────────────*/

    /* U = sqrt(w) * K @ Szz */
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                nx, nz, nz,
                sqrt(w), ukf->K, nx,
                ukf->Szz, nz,
                0.0, ukf->U, nx);

    /* Perform nz rank-1 downdates, one for each column of U */
    double *u_work = ukf->work_mat; /* Reuse workspace */

    for (int i = 0; i < nz; i++)
    {
        cholesky_downdate(
            ukf->S,
            ukf->U + i * nx,
            u_work,
            ukf->c_work,
            ukf->s_work,
            nx);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMBINED STEP
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_step(StudentT_SRUKF *restrict ukf, const double *restrict z)
{
    srukf_predict(ukf);
    srukf_update(ukf, z);
}

/**
 * @brief Process multiple measurements in one call (for Python overhead measurement)
 *
 * @param ukf       Filter instance
 * @param z_all     All measurements concatenated (n_steps * nz doubles)
 * @param n_steps   Number of steps to process
 */
void srukf_step_batch(StudentT_SRUKF *restrict ukf, const double *restrict z_all, int n_steps)
{
    const int nz = ukf->nz;
    for (int i = 0; i < n_steps; i++)
    {
        srukf_step(ukf, z_all + i * nz);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * ACCESSORS
 *───────────────────────────────────────────────────────────────────────────*/

const double *srukf_get_state(const StudentT_SRUKF *ukf)
{
    return ukf->x;
}

const double *srukf_get_sqrt_cov(const StudentT_SRUKF *ukf)
{
    return ukf->S;
}

double srukf_get_nis(const StudentT_SRUKF *ukf)
{
    return ukf->nis;
}

double srukf_get_student_weight(const StudentT_SRUKF *ukf)
{
    return ukf->student_weight;
}

double srukf_get_volatility(const StudentT_SRUKF *ukf)
{
    return exp(ukf->x[ukf->xi_index]);
}

double srukf_get_mahalanobis_sq(const StudentT_SRUKF *ukf)
{
    return ukf->mahalanobis_sq;
}

int srukf_get_nx(const StudentT_SRUKF *ukf)
{
    return ukf->nx;
}

int srukf_get_nz(const StudentT_SRUKF *ukf)
{
    return ukf->nz;
}

const double *srukf_get_predicted_state(const StudentT_SRUKF *ukf)
{
    return ukf->x_pred;
}

const double *srukf_get_predicted_measurement(const StudentT_SRUKF *ukf)
{
    return ukf->z_pred;
}

const double *srukf_get_innovation(const StudentT_SRUKF *ukf)
{
    return ukf->innovation;
}

const double *srukf_get_kalman_gain(const StudentT_SRUKF *ukf)
{
    return ukf->K;
}

/*─────────────────────────────────────────────────────────────────────────────
 * MISSING DATA HANDLING
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_predict_only(StudentT_SRUKF *restrict ukf)
{
    /* Just run predict, skip update */
    srukf_predict(ukf);

    /* Set diagnostics to indicate no update */
    ukf->nis = 0.0;
    ukf->mahalanobis_sq = 0.0;
    ukf->student_weight = 1.0;
}

int srukf_update_partial(StudentT_SRUKF *restrict ukf,
                         const double *restrict z,
                         const bool *restrict mask)
{
    const int nz = ukf->nz;

    /* Count observed components */
    int n_obs = 0;
    for (int i = 0; i < nz; i++)
    {
        if (mask[i])
            n_obs++;
    }

    /* If nothing observed, skip update */
    if (n_obs == 0)
    {
        ukf->nis = 0.0;
        ukf->mahalanobis_sq = 0.0;
        ukf->student_weight = 1.0;
        return 0;
    }

    /* If all observed, use standard update */
    if (n_obs == nz)
    {
        srukf_update(ukf, z);
        return nz;
    }

    /* Partial update: extract observed components */
    /* This is a simplified approach - we update only observed components */
    /* More sophisticated: build reduced H, R matrices dynamically */

    const int nx = ukf->nx;
    const int n_sig = ukf->n_sig;
    const int xi_idx = ukf->xi_index;
    const double nu = ukf->nu;

    /* Use X_pred from predict step */
    const double *restrict X_pred = ukf->X_pred;

    /* Build reduced measurement for observed components only */
    double z_obs[16]; /* Stack allocation for small nz */
    double H_obs[16 * 16];
    double R0_obs[16 * 16];
    int obs_idx[16];

    /* Safety check */
    if (n_obs > 16)
    {
        /* Fall back to full update if too many (shouldn't happen in trading) */
        srukf_update(ukf, z);
        return nz;
    }

    /* Extract observed indices and values */
    int k = 0;
    for (int i = 0; i < nz; i++)
    {
        if (mask[i])
        {
            obs_idx[k] = i;
            z_obs[k] = z[i];
            k++;
        }
    }

    /* Extract rows of H for observed components */
    for (int i = 0; i < n_obs; i++)
    {
        for (int j = 0; j < nx; j++)
        {
            H_obs[i + j * n_obs] = ukf->H[obs_idx[i] + j * nz];
        }
    }

    /* Extract submatrix of R0 */
    for (int i = 0; i < n_obs; i++)
    {
        for (int j = 0; j < n_obs; j++)
        {
            R0_obs[i + j * n_obs] = ukf->R0[obs_idx[i] + obs_idx[j] * nz];
        }
    }

    /* State-dependent noise scales */
    double *restrict xi_work = ukf->xi_work;
    double *restrict R_scales = ukf->R_scales;

    for (int i = 0; i < n_sig; i++)
    {
        xi_work[i] = 2.0 * X_pred[xi_idx + i * nx];
    }
    vdExp(n_sig, xi_work, R_scales);

    /* Measurement sigma points: Z_sig = H_obs @ X_pred */
    double *restrict Z_sig = ukf->work_mat;
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n_obs, n_sig, nx,
                1.0, H_obs, n_obs,
                X_pred, nx,
                0.0, Z_sig, n_obs);

    /* Predicted measurement mean */
    double z_pred_obs[16];
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                n_obs, n_sig,
                1.0, Z_sig, n_obs,
                ukf->Wm, 1,
                0.0, z_pred_obs, 1);

    /* Innovation */
    double innov[16];
    for (int i = 0; i < n_obs; i++)
    {
        innov[i] = z_obs[i] - z_pred_obs[i];
    }

    /* Innovation covariance (simplified: use mean R_scale) */
    double avg_R_scale = cblas_ddot(n_sig, ukf->Wm, 1, R_scales, 1);

    /* Build Szz for reduced dimension */
    double *restrict Z_c = ukf->work_mat + n_obs * n_sig;
    const double *restrict Wc_sqrt = ukf->Wc_sqrt;

    for (int i = 0; i < n_sig; i++)
    {
        const double sw = Wc_sqrt[i];
        for (int j = 0; j < n_obs; j++)
        {
            Z_c[j + i * n_obs] = sw * (Z_sig[j + i * n_obs] - z_pred_obs[j]);
        }
    }

    double Szz[16 * 16];
    memset(Szz, 0, sizeof(Szz));

    /* Szz = Z_c @ Z_c' */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                n_obs, n_sig,
                1.0, Z_c, n_obs,
                0.0, Szz, n_obs);

    /* Add measurement noise */
    cblas_dsyrk(CblasColMajor, CblasLower, CblasNoTrans,
                n_obs, n_obs,
                avg_R_scale, R0_obs, n_obs,
                1.0, Szz, n_obs);

    /* Cholesky */
    LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', n_obs, Szz, n_obs);

    /* Solve for v: Szz @ v = innovation */
    double v[16];
    cblas_dcopy(n_obs, innov, 1, v, 1);
    cblas_dtrsv(CblasColMajor, CblasLower, CblasNoTrans, CblasNonUnit,
                n_obs, Szz, n_obs, v, 1);

    /* Mahalanobis distance */
    double d_sq = cblas_ddot(n_obs, v, 1, v, 1);
    ukf->mahalanobis_sq = d_sq;

    /* Student-t weight */
    double w = (nu + (double)n_obs) / (nu + d_sq);
    if (w > 1.0)
        w = 1.0;
    ukf->student_weight = w;

    /* NIS */
    ukf->nis = d_sq;

    /* Cross-covariance Pxz (reduced) */
    double *restrict X_c = ukf->work_mat + 2 * n_obs * n_sig;
    const double *restrict x = ukf->x;

    for (int i = 0; i < n_sig; i++)
    {
        const double wc = ukf->Wc[i];
        for (int j = 0; j < nx; j++)
        {
            X_c[j + i * nx] = wc * (X_pred[j + i * nx] - x[j]);
        }
    }

    /* Z_centered for cross-cov */
    double *restrict Z_centered = ukf->work_mat + 2 * n_obs * n_sig + nx * n_sig;
    for (int i = 0; i < n_sig; i++)
    {
        for (int j = 0; j < n_obs; j++)
        {
            Z_centered[j + i * n_obs] = Z_sig[j + i * n_obs] - z_pred_obs[j];
        }
    }

    double Pxz[16 * 16];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasTrans,
                nx, n_obs, n_sig,
                1.0, X_c, nx,
                Z_centered, n_obs,
                0.0, Pxz, nx);

    /* Kalman gain: K = Pxz @ Szz^{-T} @ Szz^{-1} */
    double K[16 * 16];
    cblas_dcopy(nx * n_obs, Pxz, 1, K, 1);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasTrans, CblasNonUnit,
                nx, n_obs, 1.0, Szz, n_obs, K, nx);
    cblas_dtrsm(CblasColMajor, CblasRight, CblasLower, CblasNoTrans, CblasNonUnit,
                nx, n_obs, 1.0, Szz, n_obs, K, nx);

    /* State update: x = x + w * K @ innovation */
    cblas_dgemv(CblasColMajor, CblasNoTrans,
                nx, n_obs,
                w, K, nx,
                innov, 1,
                1.0, ukf->x, 1);

    /* Covariance downdate */
    double U[16 * 16];
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                nx, n_obs, n_obs,
                sqrt(w), K, nx,
                Szz, n_obs,
                0.0, U, nx);

    for (int j = 0; j < n_obs; j++)
    {
        cholesky_downdate(ukf->S, &U[j * nx], ukf->c_work, ukf->c_work, ukf->s_work, nx);
    }

    /* Update NIS tracking if enabled */
    if (ukf->nis_tracking_enabled)
    {
        int idx = ukf->nis_window_idx;
        int ws = ukf->nis_window_size;

        if (ukf->nis_window_fill == ws)
        {
            ukf->nis_sum -= ukf->nis_history[idx];
            ukf->nis_sum_sq -= ukf->nis_history[idx] * ukf->nis_history[idx];
        }

        ukf->nis_history[idx] = d_sq;
        ukf->nis_sum += d_sq;
        ukf->nis_sum_sq += d_sq * d_sq;

        ukf->nis_window_idx = (idx + 1) % ws;
        if (ukf->nis_window_fill < ws)
            ukf->nis_window_fill++;
    }

    return n_obs;
}

/*─────────────────────────────────────────────────────────────────────────────
 * COVARIANCE HEALTH & REPAIR
 *───────────────────────────────────────────────────────────────────────────*/

bool srukf_check_covariance(const StudentT_SRUKF *ukf)
{
    const int nx = ukf->nx;
    const double *restrict S = ukf->S;

    /* Check diagonal elements are positive and reasonable */
    for (int i = 0; i < nx; i++)
    {
        double diag = S[i + i * nx];
        if (diag <= 0.0 || diag > 1e10 || isnan(diag) || isinf(diag))
        {
            return false;
        }
    }

    /* Check for NaN/Inf in off-diagonal (lower triangle) */
    for (int j = 0; j < nx; j++)
    {
        for (int i = j + 1; i < nx; i++)
        {
            double val = S[i + j * nx];
            if (isnan(val) || isinf(val))
            {
                return false;
            }
        }
    }

    return true;
}

bool srukf_repair_covariance(StudentT_SRUKF *ukf, double min_diag)
{
    const int nx = ukf->nx;
    double *restrict S = ukf->S;
    bool repaired = false;

    /* Fix NaN/Inf */
    for (int j = 0; j < nx; j++)
    {
        for (int i = j; i < nx; i++)
        {
            double val = S[i + j * nx];
            if (isnan(val) || isinf(val))
            {
                S[i + j * nx] = (i == j) ? min_diag : 0.0;
                repaired = true;
            }
        }
    }

    /* Enforce minimum diagonal */
    for (int i = 0; i < nx; i++)
    {
        if (S[i + i * nx] < min_diag)
        {
            S[i + i * nx] = min_diag;
            repaired = true;
        }
    }

    /* Zero upper triangle (enforce lower triangular) */
    for (int j = 1; j < nx; j++)
    {
        for (int i = 0; i < j; i++)
        {
            S[i + j * nx] = 0.0;
        }
    }

    return repaired;
}

bool srukf_check_state_bounds(const StudentT_SRUKF *ukf, double max_abs)
{
    const int nx = ukf->nx;
    const double *restrict x = ukf->x;

    for (int i = 0; i < nx; i++)
    {
        if (isnan(x[i]) || isinf(x[i]) || fabs(x[i]) > max_abs)
        {
            return false;
        }
    }
    return true;
}

/*─────────────────────────────────────────────────────────────────────────────
 * WINDOWED NIS STATISTICS
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_enable_nis_tracking(StudentT_SRUKF *ukf, int window_size, double outlier_threshold)
{
    /* Free existing buffer if any */
    if (ukf->nis_history)
    {
        mkl_free(ukf->nis_history);
    }

    ukf->nis_window_size = window_size;
    ukf->nis_outlier_threshold = outlier_threshold;
    ukf->nis_window_idx = 0;
    ukf->nis_window_fill = 0;
    ukf->nis_sum = 0.0;
    ukf->nis_sum_sq = 0.0;

    ukf->nis_history = (double *)mkl_malloc(window_size * sizeof(double), ALIGN);
    if (ukf->nis_history)
    {
        memset(ukf->nis_history, 0, window_size * sizeof(double));
        ukf->nis_tracking_enabled = true;
    }
    else
    {
        ukf->nis_tracking_enabled = false;
    }
}

void srukf_get_nis_stats(const StudentT_SRUKF *ukf, SRUKF_NIS_Stats *stats)
{
    if (!ukf->nis_tracking_enabled || ukf->nis_window_fill == 0)
    {
        stats->mean = 0.0;
        stats->variance = 0.0;
        stats->trend = 0.0;
        stats->max_recent = 0.0;
        stats->n_outliers = 0;
        stats->window_fill = 0;
        return;
    }

    const int n = ukf->nis_window_fill;
    const int ws = ukf->nis_window_size;
    const double *hist = ukf->nis_history;
    const double threshold = ukf->nis_outlier_threshold;

    /* Mean */
    stats->mean = ukf->nis_sum / n;

    /* Variance */
    if (n > 1)
    {
        double var = (ukf->nis_sum_sq - ukf->nis_sum * ukf->nis_sum / n) / (n - 1);
        stats->variance = (var > 0.0) ? var : 0.0;
    }
    else
    {
        stats->variance = 0.0;
    }

    /* Max and outlier count */
    double max_val = 0.0;
    int outliers = 0;
    for (int i = 0; i < n; i++)
    {
        if (hist[i] > max_val)
            max_val = hist[i];
        if (hist[i] > threshold)
            outliers++;
    }
    stats->max_recent = max_val;
    stats->n_outliers = outliers;

    /* Trend: compare recent half to first half */
    if (n >= 10)
    {
        int half = n / 2;
        int start_idx = (ukf->nis_window_idx - n + ws) % ws;

        double sum_first = 0.0, sum_second = 0.0;
        for (int i = 0; i < half; i++)
        {
            sum_first += hist[(start_idx + i) % ws];
        }
        for (int i = half; i < n; i++)
        {
            sum_second += hist[(start_idx + i) % ws];
        }

        double mean_first = sum_first / half;
        double mean_second = sum_second / (n - half);
        stats->trend = mean_second - mean_first;
    }
    else
    {
        stats->trend = 0.0;
    }

    stats->window_fill = n;
}

bool srukf_nis_healthy(const StudentT_SRUKF *ukf)
{
    if (!ukf->nis_tracking_enabled || ukf->nis_window_fill < 10)
    {
        return true; /* Not enough data, assume healthy */
    }

    SRUKF_NIS_Stats stats;
    srukf_get_nis_stats(ukf, &stats);

    const double nz = (double)ukf->nz;

    /* Checks: */
    /* 1. Mean should be around nz (chi-square expectation) */
    if (stats.mean > 3.0 * nz)
        return false;

    /* 2. Trend shouldn't be strongly positive */
    if (stats.trend > 0.5 * nz)
        return false;

    /* 3. Outlier rate shouldn't be too high (>20%) */
    double outlier_rate = (double)stats.n_outliers / stats.window_fill;
    if (outlier_rate > 0.2)
        return false;

    return true;
}

/* Helper: Update NIS tracking (called from srukf_update) */
static inline void update_nis_tracking(StudentT_SRUKF *ukf, double nis)
{
    if (!ukf->nis_tracking_enabled)
        return;

    int idx = ukf->nis_window_idx;
    int ws = ukf->nis_window_size;

    /* Remove old value if buffer is full */
    if (ukf->nis_window_fill == ws)
    {
        double old_val = ukf->nis_history[idx];
        ukf->nis_sum -= old_val;
        ukf->nis_sum_sq -= old_val * old_val;
    }

    /* Add new value */
    ukf->nis_history[idx] = nis;
    ukf->nis_sum += nis;
    ukf->nis_sum_sq += nis * nis;

    /* Advance index */
    ukf->nis_window_idx = (idx + 1) % ws;
    if (ukf->nis_window_fill < ws)
    {
        ukf->nis_window_fill++;
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * STATE MANAGEMENT & SERIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void srukf_reset(StudentT_SRUKF *ukf,
                 const double *restrict x0,
                 const double *restrict S0)
{
    const int nx = ukf->nx;

    /* Reset state and covariance */
    cblas_dcopy(nx, x0, 1, ukf->x, 1);
    cblas_dcopy(nx * nx, S0, 1, ukf->S, 1);

    /* Clear diagnostics */
    ukf->nis = 0.0;
    ukf->mahalanobis_sq = 0.0;
    ukf->student_weight = 1.0;

    /* Clear NIS history */
    if (ukf->nis_tracking_enabled)
    {
        ukf->nis_window_idx = 0;
        ukf->nis_window_fill = 0;
        ukf->nis_sum = 0.0;
        ukf->nis_sum_sq = 0.0;
        memset(ukf->nis_history, 0, ukf->nis_window_size * sizeof(double));
    }
}

/* Serialization magic number and version */
#define SRUKF_MAGIC 0x55524B46 /* "URKF" */
#define SRUKF_VERSION 1

uint32_t srukf_version(void)
{
    return SRUKF_VERSION;
}

size_t srukf_serialize_size(const StudentT_SRUKF *ukf)
{
    const int nx = ukf->nx;
    const int nz = ukf->nz;

    size_t size = 0;

    /* Header */
    size += sizeof(uint32_t); /* magic */
    size += sizeof(uint32_t); /* version */
    size += sizeof(int);      /* nx */
    size += sizeof(int);      /* nz */

    /* State and covariance */
    size += nx * sizeof(double);      /* x */
    size += nx * nx * sizeof(double); /* S */

    /* Diagnostics */
    size += sizeof(double); /* nis */
    size += sizeof(double); /* mahalanobis_sq */
    size += sizeof(double); /* student_weight */

    /* NIS tracking metadata */
    size += sizeof(bool);   /* nis_tracking_enabled */
    size += sizeof(int);    /* nis_window_size */
    size += sizeof(double); /* nis_outlier_threshold */
    size += sizeof(int);    /* nis_window_idx */
    size += sizeof(int);    /* nis_window_fill */
    size += sizeof(double); /* nis_sum */
    size += sizeof(double); /* nis_sum_sq */

    /* NIS history (if enabled) */
    if (ukf->nis_tracking_enabled)
    {
        size += ukf->nis_window_size * sizeof(double);
    }

    return size;
}

size_t srukf_serialize(const StudentT_SRUKF *ukf,
                       void *buffer,
                       size_t buffer_size)
{
    size_t required = srukf_serialize_size(ukf);
    if (buffer_size < required)
    {
        return 0;
    }

    const int nx = ukf->nx;

    char *ptr = (char *)buffer;

    /* Header */
    uint32_t magic = SRUKF_MAGIC;
    uint32_t version = SRUKF_VERSION;
    memcpy(ptr, &magic, sizeof(magic));
    ptr += sizeof(magic);
    memcpy(ptr, &version, sizeof(version));
    ptr += sizeof(version);
    memcpy(ptr, &ukf->nx, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &ukf->nz, sizeof(int));
    ptr += sizeof(int);

    /* State and covariance */
    memcpy(ptr, ukf->x, nx * sizeof(double));
    ptr += nx * sizeof(double);
    memcpy(ptr, ukf->S, nx * nx * sizeof(double));
    ptr += nx * nx * sizeof(double);

    /* Diagnostics */
    memcpy(ptr, &ukf->nis, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &ukf->mahalanobis_sq, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &ukf->student_weight, sizeof(double));
    ptr += sizeof(double);

    /* NIS tracking metadata */
    memcpy(ptr, &ukf->nis_tracking_enabled, sizeof(bool));
    ptr += sizeof(bool);
    memcpy(ptr, &ukf->nis_window_size, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &ukf->nis_outlier_threshold, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &ukf->nis_window_idx, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &ukf->nis_window_fill, sizeof(int));
    ptr += sizeof(int);
    memcpy(ptr, &ukf->nis_sum, sizeof(double));
    ptr += sizeof(double);
    memcpy(ptr, &ukf->nis_sum_sq, sizeof(double));
    ptr += sizeof(double);

    /* NIS history */
    if (ukf->nis_tracking_enabled && ukf->nis_history)
    {
        memcpy(ptr, ukf->nis_history, ukf->nis_window_size * sizeof(double));
        ptr += ukf->nis_window_size * sizeof(double);
    }

    return (size_t)(ptr - (char *)buffer);
}

bool srukf_deserialize(StudentT_SRUKF *ukf,
                       const void *buffer,
                       size_t buffer_size)
{
    if (buffer_size < 4 * sizeof(uint32_t))
    {
        return false;
    }

    const char *ptr = (const char *)buffer;

    /* Check header */
    uint32_t magic, version;
    int nx_file, nz_file;

    memcpy(&magic, ptr, sizeof(magic));
    ptr += sizeof(magic);
    memcpy(&version, ptr, sizeof(version));
    ptr += sizeof(version);
    memcpy(&nx_file, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&nz_file, ptr, sizeof(int));
    ptr += sizeof(int);

    if (magic != SRUKF_MAGIC)
        return false;
    if (version != SRUKF_VERSION)
        return false;
    if (nx_file != ukf->nx || nz_file != ukf->nz)
        return false;

    const int nx = ukf->nx;

    /* State and covariance */
    memcpy(ukf->x, ptr, nx * sizeof(double));
    ptr += nx * sizeof(double);
    memcpy(ukf->S, ptr, nx * nx * sizeof(double));
    ptr += nx * nx * sizeof(double);

    /* Diagnostics */
    memcpy(&ukf->nis, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&ukf->mahalanobis_sq, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&ukf->student_weight, ptr, sizeof(double));
    ptr += sizeof(double);

    /* NIS tracking metadata */
    bool file_nis_enabled;
    int file_ws;

    memcpy(&file_nis_enabled, ptr, sizeof(bool));
    ptr += sizeof(bool);
    memcpy(&file_ws, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&ukf->nis_outlier_threshold, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&ukf->nis_window_idx, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&ukf->nis_window_fill, ptr, sizeof(int));
    ptr += sizeof(int);
    memcpy(&ukf->nis_sum, ptr, sizeof(double));
    ptr += sizeof(double);
    memcpy(&ukf->nis_sum_sq, ptr, sizeof(double));
    ptr += sizeof(double);

    /* NIS history */
    if (file_nis_enabled)
    {
        /* Ensure we have NIS tracking enabled with same window size */
        if (!ukf->nis_tracking_enabled || ukf->nis_window_size != file_ws)
        {
            srukf_enable_nis_tracking(ukf, file_ws, ukf->nis_outlier_threshold);
        }

        if (ukf->nis_history)
        {
            memcpy(ukf->nis_history, ptr, file_ws * sizeof(double));
            ptr += file_ws * sizeof(double);
        }
    }

    return true;
}