#########################
## Library Directories ##
########################

# External libraries
EIGEN_DIR = /usr/local/include/eigen3

# This library
HODLR_DIR := $(dir $(lastword $(MAKEFILE_LIST)))/header_complex

####################
## Makefile Setup ##
####################

# Get the host-name if empty
ifeq ($(host-name),)
	host-name := $(shell hostname)
endif
# Get the kernel-name if empty
ifeq ($(kernel-name),)
	kernel-name := $(shell uname -s)
endif
# Get the arch-name is empty
ifeq ($(arch-name),)
	arch-name := $(shell uname -p)
endif

# Define the C++ compiler to use
CXX := $(shell which g++) -std=c++11

# Dependency directory and flags
DEPSDIR := $(shell mkdir -p .deps; echo .deps)
# MD: Dependency as side-effect of compilation
# MF: File for output
# MP: Include phony targets
DEPSFILE = $(DEPSDIR)/$(notdir $*.d)
DEPSFLAGS = -MD -MF $(DEPSFILE) #-MP

# Define any directories containing header files
#   To include directories use -Ipath/to/files
INCLUDES += -I.
INCLUDES += -I$(EIGEN_DIR)
INCLUDES += -I$(HODLR_DIR)

# Define cxx compile flags
CXXFLAGS  = -fopenmp -funroll-loops -O3 -W -Wall -Wextra -Wno-unused-local-typedefs #-Wfatal-errors

# Define any directories containing libraries
#   To include directories use -Lpath/to/files
LDFLAGS +=

# Define any libraries to link into executable
#   To link in libraries (libXXX.so or libXXX.a) use -lXXX
LDLIBS  +=

######################
## Makefile Options ##
######################

ifeq ($(NDEBUG),1)
	CXXFLAGS += -DFMMTL_NDEBUG
endif
ifeq ($(DEBUG),1)
	CXXFLAGS += -DFMMTL_DEBUG -g -fno-inline
endif
ifeq ($(PROFILE),1)
	CXXFLAGS += -g -pg
endif
ifeq ($(LOG),1)
	CXXFLAGS += -DFMMTL_LOGGING
endif

####################
## Makefile Rules ##
####################

# Suffix replacement rules
#   $^: the name of the prereqs of the rule
#   $<: the name of the first prereq of the rule
#   $@: the name of the target of the rule
.SUFFIXES:                                        # Delete the default suffixes
.SUFFIXES: .hpp .cpp .kern .kern.cpp .kern.cu .o  # Define our suffix list

# 'make' - default rule
all: $(EXEC)

# Default rule for creating an exec of $(EXEC) from a .o file
$(EXEC): % : %.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

# Default rule for creating a .o file from a .cpp file
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(DEPSFLAGS) -c -o $@ $<

# 'make clean' - deletes all .o and temp files, exec, and dependency file
clean:
	-$(RM) *.o $(EXEC)
	$(RM) -r $(DEPSDIR)

# Define rules that do not actually generate the corresponding file
.PHONY: clean all

# Include the dependency files
-include $(wildcard $(DEPSDIR)/*.d)
