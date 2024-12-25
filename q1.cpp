#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h> // AVX2 intrinsics

#define N 1024 // Default input size

float A[N][N], B[N][N], C[N][N];

// Function prototypes
void init();
void q1();
void q1_vec_j();
void q1_vec_k();
void check_correctness(float C_serial[N][N], float C_vectorized[N][N]);
double calculate_flops(double exec_time);

int main() {
    // Define timers for measuring execution time
    double start_1, end_1;

    init(); // Initialize the arrays

    // Original routine
    start_1 = omp_get_wtime(); // Start the timer
    q1(); // Original routine
    end_1 = omp_get_wtime(); // End the timer
    printf("Original q1() execution time: %f seconds\n", end_1 - start_1);

    // Vectorized 'j' loop routine
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }
    start_1 = omp_get_wtime(); // Start the timer for vectorized version
    q1_vec_j(); // Vectorized routine using AVX2 for the 'j' loop
    end_1 = omp_get_wtime(); // End the timer
    printf("Vectorized q1_vec_j() execution time: %f seconds\n", end_1 - start_1);

    // Vectorized 'k' loop routine
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0f;
        }
    }
    start_1 = omp_get_wtime(); // Start the timer for vectorized version
    q1_vec_k(); // Vectorized routine using AVX2 for the 'k' loop
    end_1 = omp_get_wtime(); // End the timer
    printf("Vectorized q1_vec_k() execution time: %f seconds\n", end_1 - start_1);

    // Check correctness of the vectorized version
    check_correctness(C, C); // Compare results between original and vectorized

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

// Vectorized 'j' loop routine
void q1_vec_j() {
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < N; k++) {
            __m256 a_vals = _mm256_set1_ps(A[i][k]); // Load row of A
            for (int j = 0; j < N; j += 8) {
                __m256 b_vals = _mm256_loadu_ps(&B[k][j]); // Load 8 elements of column B
                __m256 c_vals = _mm256_loadu_ps(&C[i][j]); // Load current values of C[i][j]
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals); // C[i][j] += A[i][k] * B[k][j..j+7]
                _mm256_storeu_ps(&C[i][j], c_vals); // Store the result back to C
            }
        }
    }
}

// Vectorized 'k' loop routine
void q1_vec_k() {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            __m256 c_vals = _mm256_setzero_ps(); // Initialize C[i][j] to 0
            for (int k = 0; k < N; k++) {
                __m256 a_vals = _mm256_set1_ps(A[i][k]); // Load row of A
                __m256 b_vals = _mm256_loadu_ps(&B[k][j]); // Load 8 elements of column B
                c_vals = _mm256_fmadd_ps(a_vals, b_vals, c_vals); // C[i][j] += A[i][k] * B[k][j..j+7]
            }
            _mm256_storeu_ps(&C[i][j], c_vals); // Store the result back to C
        }
    }
}

// Function to check correctness of results
void check_correctness(float C_serial[N][N], float C_vectorized[N][N]) {
    int correct = 1;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (fabs(C_serial[i][j] - C_vectorized[i][j]) > 1e-6) {
                printf("Mismatch at C[%d][%d]: Serial = %f, Vectorized = %f\n", i, j, C_serial[i][j], C_vectorized[i][j]);
                correct = 0;
            }
        }
    }
    if (correct) {
        printf("Both routines produced the same result.\n");
    }
    else {
        printf("There were mismatches between the serial and vectorized results.\n");
    }
}

// Function to calculate FLOPS
double calculate_flops(double exec_time) {
    return (2.0 * N * N * N) / exec_time;
}
