#include "stdafx.h"
#include <memory>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <stdio.h>


void gram_schmidt(double* A, double* B, int N, int M) {

#pragma omp parallel shared(A, B)
  {
#pragma omp for
    for (int i = 1; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        B[i * N + j] = A[i * N + j];
      }
    }

    for (int i = 1; i < N; ++i) {
      for (int j = 0; j < i; ++j) {
        double scolar_ab = 0.0;
        double scolar_bb = 0.0;
        for (int k = 0; k < M; ++k) {
          scolar_ab += B[j * N + k] * A[i * N + k];
          scolar_bb += B[j * N + k] * B[j * N + k];
        }
        for (int k = 0; k < M; ++k) {
          B[j * N + k] -= (scolar_ab / scolar_bb) * B[j * N + k];
        }
      }
    }
  }
}

int main()
{
  int N = 300;
  int M = 800;

  double* A = (double*)malloc(M * N * sizeof(double));
  double* B = (double*)malloc(M * N * sizeof(double));
  
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = rand() - RAND_MAX / 2;
    }
  }

  printf("Threads    Speedup coeff\n");

  double start_time;
  double optimal_time = 1e100;
  double serial_part = 0;
  int optimal_processors_count;

  for (int t = 1; t <= 8; ++t) {

    omp_set_num_threads(t);

    clock_t begin = clock();

    gram_schmidt(A, B, N, M);

    clock_t end = clock();

    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    double speedup = 1;

    if (t == 1) {
      start_time = time_spent;
    }
    else {
      speedup = start_time / time_spent;
      serial_part += (t / speedup - 1) / (t - 1) / 7;
    }

    if (time_spent < optimal_time) {
      optimal_time = time_spent;
      optimal_processors_count = t;
    }

    printf("%d          %f\n", t, speedup);
  }

  free(A);
  free(B);

  printf("\nSerial part: %f%", serial_part);

  getchar();


  return 0;
}