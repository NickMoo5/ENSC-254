#include <stdio.h>
#include <stdlib.h>
#include "my_timer.h"
#include <x86intrin.h>
#include <immintrin.h>
#include <omp.h>

#define NI 4096
#define NJ 4096
#define NK 4096
#define TILE_SIZE 16

/* Array initialization. */
static
void init_array(float C[NI*NJ], float A[NI*NK], float B[NK*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i*NJ+j] = (float)((i*j+1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i*NK+j] = (float)(i*(j+1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i*NJ+j] = (float)(i*(j+2) % NJ) / NJ;
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(float C[NI*NJ])
{
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i*NJ+j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_and_valid_array_sum(float C[NI*NJ])
{
  int i, j;

  float sum = 0.0;
  float golden_sum = 27789682688.000000;
  
  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i*NJ+j];

  if ( abs(sum-golden_sum)/golden_sum > 0.00001 ) // more than 0.001% error rate
    printf("Incorrect sum of C array. Expected sum: %f, your sum: %f\n", golden_sum, sum);
  else
    printf("Correct result. Sum of C array = %f\n", sum);
}

// Function to allocate memory for a matrix
int** allocateMatrix(int numRows, int numCols) {
    int** matrix = (int**)malloc(numRows * sizeof(int*));
    for (int i = 0; i < numRows; i++) {
        matrix[i] = (int*)malloc(numCols * sizeof(int));
    }
    return matrix;
}

// Function to deallocate memfla30@ensc-hacc-22dlory for a matrix
void deallocateMatrix(int** matrix, int numRows) {
    for (int i = 0; i < numRows; i++) {
        free(matrix[i]);
    }
    free(matrix);
}


/* Main computational kernel: baseline. The whole function will be timed,
   including the call and return. DO NOT change the baseline.*/
static
void gemm_base(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i*NJ+j] *= beta;
    }
    for (j = 0; j < NJ; j++) {
      for (k = 0; k < NK; ++k) {
	      C[i*NJ+j] += alpha * A[i*NK+k] * B[k*NJ+j];
      }
    }
  }
}

/* Main computational kernel: with tiling optimization. */
static
void gemm_tile(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  int i, j, k, ii, jj;

// => Form C := alpha*A*B + beta*C,
//A is NIxNK
//B is NKxNJ
//C is NIxNJ

// Loop tiling parameters
  int tile_size = 16; // Adjust the tile size as needed, depending on cache size and matrix size.

  for (i = 0; i < NI; i += tile_size) {
    for (j = 0; j < NJ; j += tile_size) {
      // Initialize the tiles for the current iteration
      float C_tile[tile_size * tile_size] = {0.0};
      
      // Perform the matrix multiplication for the current tile
      for (k = 0; k < NK; k += tile_size) {
        for (ii = i; ii < i + tile_size; ii++) {
          for (jj = j; jj < j + tile_size; jj++) {
            for (int kk = k; kk < k + tile_size; kk++) {
              C_tile[(ii - i) * tile_size + (jj - j)] += alpha * A[ii * NK + kk] * B[kk * NJ + jj];
            }
          }
        }
      }

      // Update the main C matrix with the results from the current tile
      for (ii = i; ii < i + tile_size; ii++) {
        for (jj = j; jj < j + tile_size; jj++) {
          C[ii * NJ + jj] = beta * C[ii * NJ + jj] + C_tile[(ii - i) * tile_size + (jj - j)];
        }
      }
    }
  }

}

static inline void matrix_multiply(float *C_tile, float *A, float *B, float alpha) {
    // Load vectors for alpha multiplication
    __m256 alpha_vec = _mm256_set1_ps(alpha);

    for (int i = 0; i < TILE_SIZE; i++) {
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load B elements for multiplication
            __m256 B_vec = _mm256_broadcast_ss(&B[k * TILE_SIZE + i]);

            for (int j = 0; j < TILE_SIZE; j += 8) {
                // Load A elements for multiplication
                __m256 A_vec = _mm256_loadu_ps(&A[i * TILE_SIZE + k]);

                // Multiply: result = alpha * A[i * TILE_SIZE + k] * B[k * TILE_SIZE + j]
                __m256 result = _mm256_mul_ps(alpha_vec, _mm256_mul_ps(A_vec, B_vec));

                // Load C_tile elements for addition
                __m256 C_tile_vec = _mm256_loadu_ps(&C_tile[i * TILE_SIZE + j]);

                // Add the result to C_tile: C_tile[i * TILE_SIZE + j] += result
                C_tile_vec = _mm256_add_ps(C_tile_vec, result);

                // Store the result back to C_tile
                _mm256_storeu_ps(&C_tile[i * TILE_SIZE + j], C_tile_vec);
            }
        }
    }
}

static void gemm_tile_simds(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta) {
    int i, j, k, ii, jj;

    for (i = 0; i < NI; i += TILE_SIZE) {
        for (j = 0; j < NJ; j += TILE_SIZE) {
            // Initialize the tiles for the current iteration
            float C_tile[TILE_SIZE * TILE_SIZE] = {0.0};

            // Perform the matrix multiplication for the current tile
            for (k = 0; k < NK; k += TILE_SIZE) {
                matrix_multiply(C_tile, &A[i * NK + k], &B[k * NJ + j], alpha);
            }

            // Update the main C matrix with the results from the current tile
            for (ii = i; ii < i + TILE_SIZE; ii++) {
                for (jj = j; jj < j + TILE_SIZE; jj++) {
                    C[ii * NJ + jj] = beta * C[ii * NJ + jj] + C_tile[(ii - i) * TILE_SIZE + (jj - j)];
                }
            }
        }
    }
}


/* Main computational kernel: with tiling and simd optimizations. */
static
void gemm_tile_simd(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{
  // => Form C := alpha*A*B + beta*C,
  //A is NIxNK
  //B is NKxNJ
  //C is NIxNJ
  int i, j, k, ii, jj;

  // Load vectors for alpha multiplication
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  __m256 beta_vec = _mm256_set1_ps(beta);


  // Loop tiling parameters
  int tile_size = 16; // Adjust the tile size as needed
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j+=8) {
      // Load 8 elements from C into an AVX2 register
      __m256 c_vec = _mm256_loadu_ps(&C[i * NJ + j]);

      // Multiply the elements with beta
      c_vec = _mm256_mul_ps(c_vec, beta_vec);

      // Store the results back to C
      _mm256_storeu_ps(&C[i * NJ + j], c_vec);
    }
  }
  
  for (i = 0; i < NI; i += tile_size) {
    for (k = 0; k < NK; k += tile_size) {
      for (j = 0; j < NJ; j += tile_size) {
          for (ii = i; ii < i + tile_size; ii++) {
            for (jj = j; jj < j + tile_size; jj+=8) {
              __m256 sum = _mm256_setzero_ps();
              for (int kk = k; kk < k + tile_size; kk++) {
                //__m256 aVect = _mm256_set1_ps(A[ii * NK + kk]);
                //aVect = _mm256_mul_ps(aVect, alpha_vec);

                __m256 aVect = _mm256_set1_ps(alpha * A[ii * NK + kk]);
                __m256 bVect = _mm256_loadu_ps(&B[kk * NJ + jj]); // Load 8 elements of B
                sum = _mm256_add_ps(sum, _mm256_mul_ps(aVect, bVect));
              }
              __m256 cVect = _mm256_loadu_ps(&C[ii * NJ + jj]);
              cVect = _mm256_add_ps(cVect, sum);
              _mm256_storeu_ps(&C[ii * NJ + jj], cVect);
            }
          }
        }
      }
  }

}



/* Main computational kernel: with tiling, simd, and parallelization optimizations. */
static
void gemm_tile_simd_par(float C[NI*NJ], float A[NI*NK], float B[NK*NJ], float alpha, float beta)
{

// => Form C := alpha*A*B + beta*C,
  //A is NIxNK
  //B is NKxNJ
  //C is NIxNJ
  int i, j, k, ii, jj;

  // Load vectors for alpha multiplication
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  __m256 beta_vec = _mm256_set1_ps(beta);


  // Loop tiling parameters
  int tile_size = 16; // Adjust the tile size as needed

  #pragma omp parallel for
  for (i = 0; i < NI; i++) {
    for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }
  }
  
  #pragma omp parallel for num_threads(20)
  for (i = 0; i < NI; i += tile_size) {
    for (k = 0; k < NK; k += tile_size) {
      for (j = 0; j < NJ; j += tile_size) {
          for (ii = i; ii < i + tile_size; ii++) {
            for (jj = j; jj < j + tile_size; jj+=8) {
              __m256 sum = _mm256_setzero_ps();
              for (int kk = k; kk < k + tile_size; kk++) {

                __m256 aVect = _mm256_set1_ps(alpha * A[ii * NK + kk]);
                __m256 bVect = _mm256_loadu_ps(&B[kk * NJ + jj]); // Load 8 elements of B
                sum = _mm256_add_ps(sum, _mm256_mul_ps(aVect, bVect));
              }
              __m256 cVect = _mm256_loadu_ps(&C[ii * NJ + jj]);
              cVect = _mm256_add_ps(cVect, sum);
              _mm256_storeu_ps(&C[ii * NJ + jj], cVect);
            }
          }
        }
      }
  }

  /*#pragma omp parallel for
  for (i = 0; i < NI; i += tile_size) {
        for (k = 0; k < NK; k += tile_size) {
            for (j = 0; j < NJ; j += tile_size) {
                for (ii = i; ii < i + tile_size; ii++) {
                    for (jj = j; jj < j + tile_size; jj += simdSize) {
                        __m256 sum0 = _mm256_setzero_ps();
                        __m256 sum1 = _mm256_setzero_ps();
                        __m256 sum2 = _mm256_setzero_ps();
                        __m256 sum3 = _mm256_setzero_ps();
                        __m256 sum4 = _mm256_setzero_ps();
                        __m256 sum5 = _mm256_setzero_ps();
                        __m256 sum6 = _mm256_setzero_ps();
                        __m256 sum7 = _mm256_setzero_ps();

                        for (int kk = k; kk < k + tile_size; kk++) {
                            __m256 aVect = _mm256_set1_ps(alpha * A[ii * NK + kk]);
                            __m256 bVect0 = _mm256_loadu_ps(&B[kk * NJ + jj]);
                            __m256 bVect1 = _mm256_loadu_ps(&B[kk * NJ + jj + 8]);
                            __m256 bVect2 = _mm256_loadu_ps(&B[kk * NJ + jj + 16]);
                            __m256 bVect3 = _mm256_loadu_ps(&B[kk * NJ + jj + 24]);
                            __m256 bVect4 = _mm256_loadu_ps(&B[kk * NJ + jj + 32]);
                            __m256 bVect5 = _mm256_loadu_ps(&B[kk * NJ + jj + 40]);
                            __m256 bVect6 = _mm256_loadu_ps(&B[kk * NJ + jj + 48]);
                            __m256 bVect7 = _mm256_loadu_ps(&B[kk * NJ + jj + 56]);

                            sum0 = _mm256_add_ps(sum0, _mm256_mul_ps(aVect, bVect0));
                            sum1 = _mm256_add_ps(sum1, _mm256_mul_ps(aVect, bVect1));
                            sum2 = _mm256_add_ps(sum2, _mm256_mul_ps(aVect, bVect2));
                            sum3 = _mm256_add_ps(sum3, _mm256_mul_ps(aVect, bVect3));
                            sum4 = _mm256_add_ps(sum4, _mm256_mul_ps(aVect, bVect4));
                            sum5 = _mm256_add_ps(sum5, _mm256_mul_ps(aVect, bVect5));
                            sum6 = _mm256_add_ps(sum6, _mm256_mul_ps(aVect, bVect6));
                            sum7 = _mm256_add_ps(sum7, _mm256_mul_ps(aVect, bVect7));
                        }

                        __m256 cVect0 = _mm256_loadu_ps(&C[ii * NJ + jj]);
                        __m256 cVect1 = _mm256_loadu_ps(&C[ii * NJ + jj + 8]);
                        __m256 cVect2 = _mm256_loadu_ps(&C[ii * NJ + jj + 16]);
                        __m256 cVect3 = _mm256_loadu_ps(&C[ii * NJ + jj + 24]);
                        __m256 cVect4 = _mm256_loadu_ps(&C[ii * NJ + jj + 32]);
                        __m256 cVect5 = _mm256_loadu_ps(&C[ii * NJ + jj + 40]);
                        __m256 cVect6 = _mm256_loadu_ps(&C[ii * NJ + jj + 48]);
                        __m256 cVect7 = _mm256_loadu_ps(&C[ii * NJ + jj + 56]);

                        cVect0 = _mm256_add_ps(cVect0, sum0);
                        cVect1 = _mm256_add_ps(cVect1, sum1);
                        cVect2 = _mm256_add_ps(cVect2, sum2);
                        cVect3 = _mm256_add_ps(cVect3, sum3);
                        cVect4 = _mm256_add_ps(cVect4, sum4);
                        cVect5 = _mm256_add_ps(cVect5, sum5);
                        cVect6 = _mm256_add_ps(cVect6, sum6);
                        cVect7 = _mm256_add_ps(cVect7, sum7);

                        _mm256_storeu_ps(&C[ii * NJ + jj], cVect0);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 8], cVect1);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 16], cVect2);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 24], cVect3);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 32], cVect4);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 40], cVect5);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 48], cVect6);
                        _mm256_storeu_ps(&C[ii * NJ + jj + 56], cVect7);
                    }
                }
            }
        }
    }*/

}

int main(int argc, char** argv)
{
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI*NK*sizeof(float));
  float *B = (float *)malloc(NK*NJ*sizeof(float));
  float *C = (float *)malloc(NI*NJ*sizeof(float));

  /* opt selects which gemm version to run */
  int opt = 0;
  if(argc == 2) {
    opt = atoi(argv[1]);
  }
  //printf("option: %d\n", opt);
  
  /* Initialize array(s). */
  init_array (C, A, B);

  /* Start timer. */
  timespec timer = tic();

  switch(opt) {
  case 0: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
    break;
  case 1: // tiling
    /* Run kernel. */
    gemm_tile (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling time");
    break;
  case 2: // tiling and simd
    /* Run kernel. */
    gemm_tile_simd (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd time");
    break;
  case 3: // tiling, simd, and parallelization
    /* Run kernel. */
    gemm_tile_simd_par (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "tiling-simd-par time");
    break;
  default: // baseline
    /* Run kernel. */
    gemm_base (C, A, B, 1.5, 2.5);
    /* Stop and print timer. */
    toc(&timer, "baseline time");
  }
  /* Print results. */
  print_and_valid_array_sum(C);

  /* free memory for A, B, C */
  free(A);
  free(B);
  free(C);
  
  return 0;
}
