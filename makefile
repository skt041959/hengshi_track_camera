CC= gcc
IROOT=/home/skt/code/opencv-learning-and-transplantation/DALSA/Sapera

INC_PATH = -I$(IROOT)/include -I$(IROOT)/examples/common -I$(IROOT)/classes/basic

DEBUGFLAGS = -g 

CXX_COMPILE_OPTIONS = -x c++ -c $(DEBUGFLAGS) -DPOSIX_HOSTPC -D_REENTRANT -fno-for-scope \
			-Wall -Wno-parentheses -Wno-missing-braces \
			-Wno-unknown-pragmas -Wno-cast-qual -Wno-unused-function -Wno-unused-label

C_COMPILE_OPTIONS= -x c $(DEBUGFLAGS) -fhosted -Wall -Wno-parentheses -Wno-missing-braces \
		   	-Wno-unknown-pragmas -Wno-cast-qual -Wno-unused-function -Wno-unused-label


LCLLIBS=  -lpthread -lstdc++ -L/usr/X11R6/lib -lXext -lX11 -L /usr/local/lib -lSapera++ -lSaperaLT
OPENCV_LIBS = -lopencv_core -lopencv_features2d -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_video -lopencv_videostab
#OPENCV_LIBS = -lopencv_calib3d -lopencv_contrib -lopencv_core -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_ocl -lopencv_photo -lopencv_stitching -lopencv_superres -lopencv_ts -lopencv_video -lopencv_videostab
OPENCV_EXTRA_LIBS = /usr/lib64/libGL.so /usr/lib64/libGLU.so -lrt -lpthread -lm -ldl
#OPENCV_EXTRA_LIBS = -ltbb /lib64/libXext.so /lib64/libX11.so /lib64/libICE.so /lib64/libSM.so /lib64/libGL.so /lib64/libGLU.so -lrt -lpthread -lm -ldl

VPATH= . : $(IROOT)/examples/common

ifndef ARCH
  ARCH := $(shell uname -m | sed -e s/i.86/i386/ -e s/x86_64/x86_64/ -e s/sun4u/sparc64/ \
                -e s/arm.*/arm/ -e s/sa110/arm/)
endif

ifeq  ($(ARCH), x86_64)
	ARCH_OPTIONS= -Dx86_64
else
	ARCH_OPTIONS= -D__i386__ 
endif

%.o : %.cpp
	$(CC) -I. $(INC_PATH) $(CXX_COMPILE_OPTIONS) -D SHOOTING $(ARCH_OPTIONS) -c $< -o $@

%.o : %.c
	$(CC) -I. $(INC_PATH) $(C_COMPILE_OPTIONS) $(ARCH_OPTIONS) -c $< -o $@

OBJS= ./main.o\
	  ./camshift.o\
	  ./canny.o array.o \
	  ./moments.o

main : $(OBJS)
	$(CC) -g -o main $(OBJS) $(LCLLIBS)  $(OPENCV_LIBS) $(OPENCV_EXTRA_LIBS)

clean:
	rm *.o main 


