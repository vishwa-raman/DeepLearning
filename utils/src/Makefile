# Makefile for ADM

CC = g++
AR = ar
LD = ld
RANLIB = ranlib
RM = /bin/rm
MKDIR = /bin/mkdir
CP = /bin/cp
RM = /bin/rm

#makedepend flags
DFLAGS = 

#Compiler flags
#if mode variable is empty, setting debug build mode
ifeq ($(mode),opt)
   CFLAGS = -Wall -O3 -fPIC -shared -fopenmp
   BUILD_DIR = ../build/src.opt
else
   mode = debug
   CFLAGS = -g -Wall -fPIC -shared -DSINGLETHREADED
   BUILD_DIR = ../build/src
endif

CFILES = Globals.cpp Preprocess.cpp Annotations.cpp xmlToIDX.cpp classify.cpp

OFILES = Globals.o Preprocess.o Annotations.o xmlToIDX.o classify.o

TRAIN_DATA_OFILES = $(BUILD_DIR)/Globals.o $(BUILD_DIR)/Preprocess.o $(BUILD_DIR)/Annotations.o

INSTALL_DIR = ../../install/lib
HEADER_DIR = ../../install/include
BIN_DIR = ../../install/bin

TRAIN_DATA_LIB = $(INSTALL_DIR)/libdata.a
XML_TO_IDX = $(BIN_DIR)/xmlToIDX
CLASSIFY = $(BIN_DIR)/classify
TRAIN_DATA_INCLUDE = Annotations.h Preprocess.h

OUT = $(TRAIN_DATA_LIB) $(XML_TO_IDX)

INCLUDES = -I ./ `pkg-config opencv --cflags` `pkg-config fftw3 --cflags`
LIBS = `pkg-config opencv --cflags --libs` `pkg-config fftw3 --cflags --libs`

OBJS := $(patsubst %.cpp, $(BUILD_DIR)/%.o, $(filter %.cpp,$(CFILES)))

#OBJS = $(patsubst %,$(BUILD_DIR)/%,$(OFILES))

.phony:all header

all: information $(OUT)

information:
ifneq ($(mode),opt)
ifneq ($(mode),debug)
	@echo "Invalid build mode." 
	@echo "Please use 'make mode=opt' or 'make mode=debug'"
	@exit 1
endif
endif
	@echo "Building on "$(mode)" mode"
	@echo ".........................."

$(BUILD_DIR)/%.o: %.cpp $(TRAIN_DATA_INCLUDE)
	$(MKDIR) -p $(BUILD_DIR)	
	$(CC) -c $(INCLUDES) -o $@ $< $(CFLAGS) 

$(OUT): $(OBJS) 
	$(MKDIR) -p $(INSTALL_DIR)	
	$(MKDIR) -p $(BIN_DIR)	
	$(MKDIR) -p $(HEADER_DIR)	
	$(AR) rcs $(TRAIN_DATA_LIB) $(TRAIN_DATA_OFILES)
	$(RANLIB) $(TRAIN_DATA_LIB)
	$(CP) -p $(TRAIN_DATA_INCLUDE) $(HEADER_DIR)
	$(CC) -o $(XML_TO_IDX) $(BUILD_DIR)/xmlToIDX.o -L$(INSTALL_DIR) -ldata $(LIBS)
	$(CC) -o $(CLASSIFY) $(BUILD_DIR)/classify.o -L$(INSTALL_DIR) -ldata $(LIBS)
	@echo train_utils finished

header:
	$(MKDIR) -p $(HEADER_DIR)	
	$(CP) -p $(TRAIN_DATA_INCLUDE) $(HEADER_DIR)
	@echo header finished

depend:
	makedepend -- $(DFLAGS) -- $(CFILES)

.PHONY: clean

clean:
	$(RM) -f $(BUILD_DIR)/*.o $(OUT)

