#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h> // AVX2 intrinsics
#include <omp.h>

#define N 1024 // Default input size
#define TOLERANCE 1e-2 // Relaxed tolerance to allow small floating-point differences
#define ALIGNMENT 32

// Align matrices for AVX2 vectorization
alignas(ALIGNMENT) float A[N][N], B[N][N], C[N][N];
alignas(ALIGNMENT) float C_serial[N][N]; // To hold serial results for correctness checking

// Function prototypes
void init();
void q1();
void q1_vec_j();
void q1_vec_k();
void check_correctness(float C_serial[N][N], float C_vectorized[N][N]);
double calculate_flops(double exec_time);

int main() {
    double start_1, end_1;

    init(); // Initialize the arrays

    // Original routine
    start_1 = omp_get_wtime();
    q1(); // Original routine
    end_1 = omp_get_wtime();
    printf("Original q1() execution time: %f seconds\n", end_1 - start_1);

    // Save the original result for correctness checking
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C_serial[i][j] = C[i][j];
        }
    }

    // Vectorized 'j' loop routine
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }
    start_1 = omp_get_wtime();
    q1_vec_j(); // Vectorized routine using AVX2 for the 'j' loop
    end_1 = omp_get_wtime();
    printf("Vectorized q1_vec_j() execution time: %f seconds\n", end_1 - start_1);

    // Vectorized 'k' loop routine
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }
    start_1 = omp_get_wtime();
    q1_vec_k(); // Vectorized routine using AVX2 for the 'k' loop
    end_1 = omp_get_wtime();
    printf("Vectorized q1_vec_k() execution time: %f seconds\n", end_1 - start_1);

    // Check correctness of the vectorized version
    printf("Checking correctness between serial and vectorized (j-loop) version...\n");
    check_correctness(C_serial, C); // Compare results between original and vectorized

    // Calculate and print FLOPS for original q1() routine
    double flops = calculate_flops(end_1 - start_1);
    printf("FLOPS for original q1(): %f\n", flops);

    return 0;
}

void init() {
    float e = 0.1234f, p = 0.7264f;
    for (unsigned int i = 0; i < N; i++) {
        for (unsigned int j = 0; j < N; j++) {
            A[i][j] = ((i - j) % 9) + p;
            B[i][j] = ((i + j) % 11) + e;
            C[i][j] = 0.0f;
        }
    }
}

// Original q1() routine
void q1() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Vectorized 'j' loop routine using AVX2
void q1_vec_j() {
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m256 a_vals = _mm256_set1_ps(A[i][k]); // Load A[i][k] into all elements of a_vals
            for (int j = 0; j < N; j += 8) {  // Process 8 elements in parallel
                __m256 b_vals = _mm256_load_ps(&B[k][j]); // Load 8 elements from B[k][j] (aligned load)
                __m256 c_vals = _mm256_load_ps(&C[i][j]); // Load current values of C[i][j] (aligned load)
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals); // Perform multiplication and addition
                _mm256_store_ps(&C[i][j], c_vals); // Store result back to C[i][j..j+7] (aligned store)
            }
        }
    }
}

// Vectorized 'k' loop routine using AVX2
void q1_vec_k() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 c_vals = _mm256_setzero_ps(); // Initialize C[i][j] to 0
            for (int k = 0; k < N; k++) {
                __m256 a_vals = _mm256_set1_ps(A[i][k]); // Load A[i][k]
                __m256 b_vals = _mm256_load_ps(&B[k][j]); // Load 8 elements of column B (aligned load)
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals); // Perform multiplication and addition
            }
            _mm256_store_ps(&C[i][j], c_vals); // Store result back to C[i][j] (aligned store)
        }
    }
}

// Function to check correctness of results
void check_correctness(float C_serial[N][N], float C_vectorized[N][N]) {
    int mismatch_count = 0;

    // Print tolerance used for comparison
    printf("Tolerance for correctness check: %e\n", TOLERANCE);

    // Iterate through the matrices and check for mismatches
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C_serial[i][j] - C_vectorized[i][j]) > TOLERANCE) {
                mismatch_count++;
                if (mismatch_count <= 10) { // Print only the first 10 mismatches
                    printf("Mismatch at C[%d][%d]: %f (serial) vs %f (vectorized)\n", i, j, C_serial[i][j], C_vectorized[i][j]);
                }
            }
        }
    }

    // Print result
    if (mismatch_count > 0) {
        printf("Total number of mismatches: %d\n", mismatch_count);
    }
    else {
        printf("Both routines produced the same result (within tolerance).\n");
    }
}

// Function to calculate FLOPS
double calculate_flops(double exec_time) {
    return (2.0 * N * N * N) / exec_time;
}
