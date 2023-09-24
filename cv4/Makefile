.SUFFIXES: .cu
PROGRAM	= main
OBJS	= main.o MyCamera.o

SRCS	= $(OBJS:%.o=%.cu)
NVCC	= nvcc
INCLUDE := /usr/local/include/opencv4/
LFLAGS	= -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

$(PROGRAM):	$(OBJS)
		$(NVCC) -g -o $(PROGRAM) $^ -I$(INCLUDE) $(LFLAGS)

.cu.o:
		$(NVCC) -c $< -I$(INCLUDE)

test:		$(PROGRAM)
		./main

gdb:		$(PROGRAM)
		gdb main

clean:;		rm -rf *.o *~ main
