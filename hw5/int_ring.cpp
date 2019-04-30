#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_ring(long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);
  int size;
  MPI_Comm_size(comm, &size);

  int* msg = (int*) malloc(Nsize*sizeof(int));
//  #pragma omp parallel for schedule(static) 
  for (long i = 0; i < Nsize; i++) msg[i] = 0;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (rank != 0) {
      MPI_Recv(msg, Nsize, MPI_INT, rank-1, repeat, comm, &status);
//      #pragma omp parallel for schedule(static) 
//      for (long i = 0; i < Nsize; i++) msg[i] = msg[i] + rank;
      msg[0] = msg[0] + rank;
      MPI_Send(msg, Nsize, MPI_INT, (rank+1)%size, repeat, comm);
    }
    else {
      MPI_Send(msg, Nsize, MPI_INT, 1, repeat, comm);
      MPI_Recv(msg, Nsize, MPI_INT, size-1, repeat, comm, &status);
    }
  }
  tt = MPI_Wtime() - tt;

  long check = Nrepeat*(size-1)*size/2;
  if (!rank) printf("After %d loops, result is %d, which should be %d\n", Nrepeat, msg[0], check);
  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  long Nrepeat = 1000;
  double tt = time_ring(Nrepeat, 1, comm);
  if (!rank) printf("ring latency: %e ms\n", tt);

  Nrepeat = 1000;
  long Nsize = 300000;
  tt = time_ring(Nrepeat, Nsize, comm);
  if (!rank) printf("ring bandwidth: %e GB/s\n", (Nsize*sizeof(int)*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
