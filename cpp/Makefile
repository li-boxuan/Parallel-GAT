.PHONY: clean

seq_gat: main.cpp node.h param.h sparse.h gat.h
	g++ -std=c++11 -o seq_gat main.cpp -fopenmp -O3 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas

omp_gat: main.cpp node.h param.h sparse.h gat.h
	g++ -std=c++11 -o omp_gat main.cpp -fopenmp -O3 -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unknown-pragmas -DUSEOPENMP

clean:
	/bin/rm -rf *~ *.o seq_gat omp_gat *.class
