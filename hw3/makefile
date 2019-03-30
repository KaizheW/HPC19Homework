# make -j # multi cores

FLAGS = -std=c++11 -O3 -march=native -fopenmp
PROGS = fast-sin omp-scan

all: $(PROGS)

fast-sin: fast-sin.cpp
	g++ $(FLAGS) -fno-tree-vectorize fast-sin.cpp -o fast-sin

omp-scan: omp-scan.cpp
	g++ $(FLAGS) omp-scan.cpp -o omp-scan

clean:
	rm -rf $(PROGS)
