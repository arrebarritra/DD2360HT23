# From https://x.momo86.net/?p=29

DEVICE ?= CPU
DEBUG ?= 0

CXX=g++
CXXFLAGS=-std=c++11 -I./include -O3 -g -Xcompiler -Wall -D$(DEVICE)

ifeq ($(DEBUG), 1)
    CXXFLAGS += -DDEBUG -g -G
endif

NVCC=nvcc
ARCH=sm_61
NVCCFLAGS= -I./include -arch=$(ARCH) -std=c++11 -O3 -g -Xcompiler -Wall --compiler-bindir=$(CXX) -D$(DEVICE)

ifeq ($(DEBUG), 1)
    NVCCFLAGS += -DDEBUG -g -G
endif

SRCDIR=src
SRCS=$(shell find $(SRCDIR) -name '*.cu' -o -name '*.cpp')

OBJDIR=bin/$(DEVICE)/DEBUG$(DEBUG)
OBJS=$(subst $(SRCDIR),$(OBJDIR), $(SRCS))
OBJS:=$(subst .cpp,.o,$(OBJS))
OBJS:=$(subst .cu,.o,$(OBJS))

BIN := ./$(OBJDIR)
TARGET=sputniPIC.out

all: dir $(BIN)/$(TARGET)

dir: ${BIN}
  
${BIN}:
	mkdir -p $(BIN)
	
$(OBJDIR):
	mkdir -p $(OBJDIR)

$(BIN)/$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $+ -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cu
	$(NVCC) $(NVCCFLAGS) $< -c -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	[ -d $(OBJDIR) ] || mkdir -p $(OBJDIR)
	$(NVCC) $(CXXFLAGS) $< -c -o $@

clean:
	rm -rf $(OBJS)
	rm -rf $(OBJDIR)/$(TARGET)
