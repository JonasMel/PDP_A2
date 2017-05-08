CC         =  mpicc
CCFLAGS    =  -O3 -march=native
CCGFLAGS   =  -g
LIBS       =  -lmpi -lm

BINS= assign2

assign2: Assign2_v5.c
	$(CC) $(CCFLAGS) -o $@ $^ $(LIBS)

clean:
	$(RM) $(BINS)



