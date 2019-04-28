#include <stdio.h>
#include <cstdlib>
#include <mpi.h>
#include <iostream>

double time_ring(int proc0, int proc1, int proc2, long Nrepeat, long Nsize, MPI_Comm comm) {
  int rank;
  MPI_Comm_rank(comm, &rank);

  char* msg = (char*) malloc(Nsize);
  for (long i = 0; i < Nsize; i++) msg[i] = 42;

  MPI_Barrier(comm);
  double tt = MPI_Wtime();
  for (long repeat  = 0; repeat < Nrepeat; repeat++) {
    MPI_Status status;
    if (repeat % 3 == 0) { // even iterations

      if (rank == proc0)
        MPI_Send(msg, Nsize, MPI_CHAR, proc1, repeat, comm);
      else if (rank == proc1)
        MPI_Recv(msg, Nsize, MPI_CHAR, proc0, repeat, comm, &status);

    }
    else if (repeat % 3 == 1) { // odd iterations

      if (rank == proc1)
        MPI_Send(msg, Nsize, MPI_CHAR, proc0, repeat, comm);
      else if (rank == proc2)
        MPI_Recv(msg, Nsize, MPI_CHAR, proc1, repeat, comm, &status);

    }
    else {

      if (rank == proc2)
        MPI_Send(msg, Nsize, MPI_CHAR, proc0, repeat, comm);
      else if (rank == proc0)
        MPI_Recv(msg, Nsize, MPI_CHAR, proc1, repeat, comm, &status);

    }
  }
  tt = MPI_Wtime() - tt;

  free(msg);
  return tt;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  if (argc < 4) {
    printf("Usage: mpirun ./int_ring <process-rank0> <process-rank1> <process-rank2>\n");
    abort();
  }
  int proc0 = atoi(argv[1]);
  int proc1 = atoi(argv[2]);
  int proc2 = atoi(argv[3]);

  int rank;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);

  long Nrepeat = 1000;
  double tt = time_ring(proc0, proc1, proc2, Nrepeat, 1, comm);
  if (!rank) printf("ring latency: %e ms\n", tt);

  Nrepeat = 10000;
  long Nsize = 1000000;
  tt = time_ring(proc0, proc1, proc2, Nrepeat, Nsize, comm);
  if (!rank) printf("ring bandwidth: %e GB/s\n", (Nsize*Nrepeat)/tt/1e9);

  MPI_Finalize();
}
