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
// => Form C := alpha*A*B + beta*C,
  //A is NIxNK
  //B is NKxNJ
  //C is NIxNJ
  int i, j, k, ii, jj, kk;
  float sum;

  // Loop tiling parameters
  int tile_size = 16; 
  
  for (i = 0; i < NI; i += tile_size) {
    for (j = 0; j < NJ; j += tile_size) {
      float C_tile[tile_size * tile_size] = {0.0};
      
      for (k = 0; k < NK; k += tile_size) {
        for (ii = i; ii < i + tile_size; ii++) {
          for (jj = j; jj < j + tile_size; jj++) {
            for (kk = k; kk < k + tile_size; kk++) {
              C_tile[(ii - i) * tile_size + (jj - j)] += alpha * A[ii * NK + kk] * B[kk * NJ + jj];
            }
          }
        }
      }

      for (ii = i; ii < i + tile_size; ii++) {
        for (jj = j; jj < j + tile_size; jj++) {
          C[ii * NJ + jj] = beta * C[ii * NJ + jj] + C_tile[(ii - i) * tile_size + (jj - j)];
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
  int a, i, j, k, ii, jj, kk;

  // Load vectors 
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  __m256 beta_vec = _mm256_set1_ps(beta);


  // Loop tiling parameters
  int tile_size = 16; 
  
  for (i = 0; i < NI; i += tile_size) {
    for (a = 0; a < tile_size; a++) {
      for (j = 0; j < NJ; j+=8) {
        __m256 cVec = _mm256_loadu_ps(&C[(i+a)*NJ+j]);
        cVec = _mm256_mul_ps(cVec, beta_vec);
        _mm256_storeu_ps(&C[(i+a)*NJ+j], cVec);
      }
    }
    
    for (k = 0; k < NK; k += tile_size) {
      for (j = 0; j < NJ; j += tile_size) {
          for (ii = i; ii < i + tile_size; ii++) {
            for (jj = j; jj < j + tile_size; jj+=8) {
              __m256 sum = _mm256_setzero_ps();
              for (kk = k; kk < k + tile_size; kk++) {

                __m256 aVect = _mm256_set1_ps(alpha * A[ii * NK + kk]);
                __m256 bVect = _mm256_loadu_ps(&B[kk * NJ + jj]); 
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
  int a, i, j, k, ii, jj, kk;

  // Load vectors 
  __m256 alpha_vec = _mm256_set1_ps(alpha);

  __m256 beta_vec = _mm256_set1_ps(beta);


  // Loop tiling parameters
  int tile_size = 16; 
  
  #pragma omp parallel for num_threads(16)
  for (i = 0; i < NI; i += tile_size) {
    for (a = 0; a < tile_size; a++) {
      for (j = 0; j < NJ; j+=8) {
        __m256 cVec = _mm256_loadu_ps(&C[(i+a)*NJ+j]);
        cVec = _mm256_mul_ps(cVec, beta_vec);
        _mm256_storeu_ps(&C[(i+a)*NJ+j], cVec);
      }
    }
    
    for (k = 0; k < NK; k += tile_size) {
      for (j = 0; j < NJ; j += tile_size) {
          for (ii = i; ii < i + tile_size; ii++) {
            for (jj = j; jj < j + tile_size; jj+=8) {
              __m256 sum = _mm256_setzero_ps();
              for (kk = k; kk < k + tile_size; kk++) {

                __m256 aVect = _mm256_set1_ps(alpha * A[ii * NK + kk]);
                __m256 bVect = _mm256_loadu_ps(&B[kk * NJ + jj]); 
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
