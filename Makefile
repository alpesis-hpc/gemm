# ----------------------------------------------------------------------------------------------

INC_DIR = inc
GEMM_DIR = gemm
TESTS_DIR = tests

BUILD_DIR = _build

# ----------------------------------------------------------------------------------------------

CC = gcc
CFLAGS_OPTS_COMPILE = -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8
CFLAGS_OPTS_GEMM = -DASMNAME=sgemm_beta \
                   -DASMFNAME=sgemm_beta_ \
                   -DNAME=sgemm_beta_ \
                   -DCNAME=sgemm_beta \
                   -DCHAR_NAME=\"sgemm_beta_\" \
                   -DCHAR_CNAME=\"sgemm_beta\" \
                   -DNO_AFFINITY
CFLAGS = -O3 -w -I$(INC_DIR) 

# ----------------------------------------------------------------------------------------------

LIB_GEMM = libgemm.a


ASM_SOURCES := $(wildcard $(GEMM_DIR)/*.S)
ASM_OBJECTS := $(patsubst %, $(BUILD_DIR)/%, $(notdir $(ASM_SOURCES:.S=.o)))
asm: $(ASM_OBJECTS)
$(BUILD_DIR)/%.o : $(GEMM_DIR)/%.S
	$(CC) $(CFLAGS) $(CFLAGS_OPTS_COMPILE) -c -UDOUBLE -UCOMPLEX $< -o $@

GEMM_SOURCES := $(wildcard $(GEMM_DIR)/*.c)
GEMM_OBJECTS := $(patsubst %, $(BUILD_DIR)/%, $(notdir $(GEMM_SOURCES:.c=.o)))
gemm: $(GEMM_OBJECTS)
$(BUILD_DIR)/%.o : $(GEMM_DIR)/%.c
	$(CC) $(CFLAGS) $(CFLAGS_OPTS_COMPILE) -c -UDOUBLE -UCOMPLEX $< -o $@

lib: asm gemm
	ar -ru $(BUILD_DIR)/$(LIB_GEMM) $(ASM_OBJECTS) $(GEMM_OBJECTS)
	ranlib $(BUILD_DIR)/$(LIB_GEMM)


# ----------------------------------------------------------------------------------------------

all: clean build lib
	$(CC) $(CFLAGS) -o $(BUILD_DIR)/test $(TESTS_DIR)/thread_level3.c -pthread -static -L$(BUILD_DIR) -lgemm

build:
	mkdir _build

clean:
	rm -rf _build
