// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int rank, p;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  // Number of random numbers per processor (this should be increased
  // for actual tests or could be passed in through the command line
  int N = 10000;

  int* vec = (int*)malloc(N*sizeof(int));
  // seed random number generator differently on every core
  srand((unsigned int) (rank + 393919));

  // fill vector with random integers
  for (int i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  double tt = MPI_Wtime();
  // sort locally
  std::sort(vec, vec+N);

  // sample p-1 entries from vector as the local splitters, i.e.,
  // every N/P-th entry of the sorted vector
  int* lsplits = (int*)malloc((p-1)*sizeof(int));
  for (int i = 0; i < p-1; i++) {
	lsplits[i] = vec[i*(N/(p-1))+N/p];
	//printf("rank: %d, splitter %d: %d\n", rank, i, lsplits[i]);
  }

  // every process communicates the selected entries to the root
  // process; use for instance an MPI_Gather
  int* splitters = NULL;
  if (rank == 0){
	splitters = (int*)malloc(sizeof(int) * (p-1)*p);
  }

  MPI_Gather(lsplits, p-1, MPI_INT, splitters, p-1, MPI_INT, 0, MPI_COMM_WORLD);

  /*if (rank==0){
        for (int i=0; i<(p-1)*p; i++){
  	 printf("splitter: %d, number: %d\n", i, splitters[i]);
	}
  }*/
  // root process does a sort and picks (p-1) splitters (from the
  // p(p-1) received elements)
  int * gsplits = (int*)malloc((p-1)*sizeof(int));
  if (rank == 0){
	std::sort(splitters,splitters+(p-1)*p);
	for (int i = 0; i < p-1; i++) {
	 gsplits[i] = splitters[i*p];
  	}
  }

  // root process broadcasts splitters to all other processes
  MPI_Bcast(gsplits, p-1, MPI_INT, 0, MPI_COMM_WORLD);

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
  int* sdispls = (int*)malloc((p)*sizeof(int));
  int* numpp = (int*)malloc((p)*sizeof(int)); // number going to each process
  sdispls[0]=0;
  for (int i=0; i < p-1; i++){
	sdispls[i+1]=std::lower_bound(vec, vec+N, gsplits[i])-vec;
	numpp[i] = sdispls[i+1]-sdispls[i];
	//if (i == 0){
	//	printf("Rank %d number going to rank 0: %d \n", rank, numpp[0]);
	//}
  }
  numpp[p-1]=N-sdispls[p-1];
  /*for (int i=0; i < p; i++){
	if (i == 0){
		printf("Rank %d number going to rank p: %d \n", rank, numpp[p-1]);
	}
  }*/
  // send and receive: first use an MPI_Alltoall to share with every
  // process how many integers it should expect, and then use
  int * nbyP = (int*)malloc((p)*sizeof(int)); // number COMING from each process
  MPI_Alltoall(numpp, 1, MPI_INT,nbyP, 1, MPI_INT,MPI_COMM_WORLD);
  int * rdispls = (int*) malloc(p*sizeof(int));
  rdispls[0]=0;
  for (int i=1; i < p; i++){
	rdispls[i]=rdispls[i-1]+nbyP[i-1];
	//if (rank == 0){
	//	printf("Rank 0 number coming from %d: %d , recieve point %d \n", i, nbyP[i], rdispls[i]);
	//}
  }
  // MPI_Alltoallv to exchange the data
  // Declare array of the right size
  int locsize = rdispls[p-1]+nbyP[p-1];
  int * sortedloc = (int*)malloc(locsize*sizeof(int));
  MPI_Alltoallv(vec, numpp, sdispls, MPI_INT, sortedloc, nbyP, rdispls, MPI_INT, MPI_COMM_WORLD);
  // do a local sort of the received data
  std::sort(sortedloc,sortedloc+locsize);
  MPI_Barrier(MPI_COMM_WORLD);
  double elapsed = MPI_Wtime() - tt;
  if (0 == rank) {
    printf("Time elapsed is %f seconds.\n", elapsed);
  }

  printf("SORTED rank: %d, first entry: %d, last entry %d \n", rank, sortedloc[0], sortedloc[locsize-1]);
  // every process writes its result to a file
  char str[20];
  sprintf(str, "FromRank_%d.txt", rank);
  FILE *f = fopen(str, "w");
  for (int i=0; i < locsize; i++){
  	fprintf(f, "%d \n", sortedloc[i]);
  }
  fclose(f);
  free(vec);
  free(splitters);
  free(lsplits);
  free(gsplits);
  free(sdispls);
  free(numpp);
  free(nbyP);
  free(rdispls);
  free(sortedloc);
  MPI_Finalize();
  return 0;
}
