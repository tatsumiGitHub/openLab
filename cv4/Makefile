PROGRAM	= main
OBJS	= main.o MyCamera.o

SRCS	= $(OBJS:%.o=%.cpp)
CC		= g++
CCFLAGS	= -I/usr/local/include/opencv4/
LFLAGS	= -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio

$(PROGRAM):	$(OBJS)
		$(CC) -g -o $(PROGRAM) $^ $(CCFLAGS) $(LFLAGS)

.cpp.o:
		$(CC) -c $< $(CCFLAGS)

test:		$(PROGRAM)
		./main

gdb:		$(PROGRAM)
		gdb main

clean:;		rm -rf *.o *~ main
