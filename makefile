# SERIAL VERSION

FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

nbody: nbody.o compute.o
	gcc $(FLAGS) $^ -o $@ $(LIBS)
nbody.o: nbody.c planets.h config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 
compute.o: compute.c config.h vector.h $(ALWAYS_REBUILD)
	gcc $(FLAGS) -c $< 



# CUDA VERSION

NVCC = nvcc

nbody_cuda: main.cu compute.cu planets.h config.h vector.h $(ALWAYS_REBUILD)
	$(NVCC) -DDEBUG main.cu compute.cu -o nbody_cuda $(LIBS)


# CLEAN

clean:
	rm -f *.o nbody nbody_cuda
