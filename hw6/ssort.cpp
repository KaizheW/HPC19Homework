// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm comm = MPI_COMM_WORLD;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 100;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  // sort locally
  double t = MPI_Wtime();
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int * lsp = (int *) malloc(sizeof(int)*(p-1));
  for (int i=0; i<p-1; i++) {
    lsp[i] = vec[i*N/(p-1)+N/p];
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  // spc: SPlitter Candidate;
  int * spc = NULL;
  if (rank == 0) {
    int* spc = (int*) malloc(sizeof(int)*(p-1)*p);
  }
  MPI_Gather(lsp, p-1, MPI_INT, spc, p-1, MPI_INT, 0, comm);

  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int * sp = (int*) malloc(sizeof(int)*(p-1));
  if (rank == 0) {
    // int* sp = (int*) malloc(sizeof(int)*(p-1));
    std::sort(spc, spc+(p-1)*p);
    for (int i=0; i<p-1; i++) sp[i] = spc[(i+1)*p-1];
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(sp, p-1, MPI_INT, 0, comm);

  // every process uses the obtained splitters to decide which
  // integers need to be sent to which other process (local bins).
  // Note that the vector is already locally sorted and so are the
  // splitters; therefore, we can use std::lower_bound function to
  // determine the bins efficiently.
  //
  // Hint: the MPI_Alltoallv exchange in the next step requires
  // send-counts and send-displacements to each process. Determining the
  // bins for an already sorted array just means to determine these
  // counts and displacements. For a splitter s[i], the corresponding
  // send-displacement for the message to process (i+1) is then given by,
  // sdispls[i+1] = std::lower_bound(vec, vec+N, s[i]) - vec;
  int* sdispls = (int*) calloc(sizeof(int), p);
  for (int i=0; i<p-1; i++) {
    sdispls[i+1] = std::lower_bound(vec, vec+N, sp[i]) - vec;
  }

  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  // MPI_Alltoallv to exchange the data
  int* nsend = (int*) malloc(p*sizeof(int));
  int* nrecv = (int*) malloc(p*sizeof(int));
  for (int i = 0; i < p-1; i++){
    nsend[i] = sdispls[i+1] - sdispls[i];
  }
  nsend[p-1] = N - sdispls[p-1];
  MPI_Alltoall(nsend, 1, MPI_INT, nrecv, 1, MPI_INT, comm);
  int* rdispls = (int*) calloc(sizeof(int), p);
  for (int i=0; i<p-1; i++) {
    rdispls[i+1] = rdispls[i] + nrecv[i];
  }
  int localsize = rdispls[p-1]+nrecv[p-1];
  int* localsort = (int*) malloc(sizeof(int)*localsize);
  MPI_Alltoallv(vec, nsend, sdispls, MPI_INT, localsort, nrecv, rdispls, MPI_INT, comm);

  // do a local sort of the received data
  std::sort(localsort, localsort+localsize);
  t = MPI_Wtime() - t;
  if (rank == 0) printf("time: %f \n", t);

  // every process writes its result to a file
  FILE* fd = NULL;
  char filename[256];
  snprintf(filename, 256, "output%02d.txt", rank);
  fd = fopen(filename, "w+");

  if(NULL == fd) {
    printf("Error opening file \n");
    return 1;
  }

  for (int i = 0; i < localsize; i++) {
    fprintf(fd, "%d \n", localsort[i]);
  }
  fclose(fd);

  free(vec);
  free(lsp);
  free(spc);
  free(sp);
  free(sdispls);
  free(rdispls);
  free(nsend);
  free(nrecv);
  free(localsort);
  MPI_Finalize();
  return 0;
}
