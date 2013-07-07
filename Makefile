################################################################################
#
# Build script for project
#
################################################################################

# Add source files here
EXECUTABLE	:= trap 
# Cuda source files (compiled with cudacc)
CUFILES_sm_13		:= trap.cu
# C/C++ source files (compiled with gcc / c++)
CCFILES		:= trap_gold.cpp 

################################################################################
# Rules and targets

include /usr/local/share/CUDA_SDK/C/common/common.mk
