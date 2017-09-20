BUILD_DIR = _build
PROJECT_DIR = src
INCLUDE_DIR = inc

CC = gcc
CFLAGS = -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=12 -I. -UDOUBLE  -UCOMPLEX 


# -------------------------------------------------------------------------------

DRIVER_DIR = $(PROJECT_DIR)/driver/
LAPACK_DIR = $(PROJECT_DIR)/lapack/
KERNEL_DIR = $(PROJECT_DIR)/kernel/

# -------------------------------------------------------------------------------

DIRVER_SOURCES := $(wildcard $(DRIVER_DIR)/*.c)
DRIVER_OBJECTS := $(patsubst %, $(BUILD_DIR)/%, $(notdir $(DIRVER_SOURCES:.c=.o)))

driver: $(DRIVER_OBJECTS)

$(BUILD_DIR)/%.o : $(DRIVER_DIR)/%.c
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@


# -------------------------------------------------------------------------------

KERNEL_SOURCES := $(wildcard $(KERNEL_DIR)/*.S)
KERNEL_OBJECTS := $(patsubst %, $(BUILD_DIR)/%, $(notdir $(KERNEL_SOURCES:.S=.o)))

kernel: $(KERNEL_OBJECTS)

$(BUILD_DIR)/%.o : $(KERNEL_DIR)/%.S
	@echo "$(RED)Compiling $< $(NC)"
	$(CC) -c $< -o $@ -I$(INCLUDE_DIR)

# -------------------------------------------------------------------------------

all: clean build
	$(CC) $(CFLAGS) $(PROJECT_DIR)/gemm.c $(DRIVER_DIR)/*.c $(KERNEL_DIR)/*.S -I$(INCLUDE_DIR) -I$(DRIVER_DIR) -o $(BUILD_DIR)/gemm


build:
	mkdir _build

clean:
	rm -rf _build
