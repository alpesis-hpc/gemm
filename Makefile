INC_DIR = inc
GEMM_DIR = gemm
SCHEDULER_DIR = scheduler
UTILS_DIR = utils
TESTS_DIR = tests
BUILD_DIR = _build



all: clean build 
	gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_beta -DASMFNAME=sgemm_beta_ -DNAME=sgemm_beta_ -DCNAME=sgemm_beta -DCHAR_NAME=\"sgemm_beta_\" -DCHAR_CNAME=\"sgemm_beta\" -DNO_AFFINITY -I$(INC_DIR) -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./$(GEMM_DIR)/gemm_beta.S -o $(BUILD_DIR)/sgemm_beta.o
	gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_itcopy -DASMFNAME=sgemm_itcopy_ -DNAME=sgemm_itcopy_ -DCNAME=sgemm_itcopy -DCHAR_NAME=\"sgemm_itcopy_\" -DCHAR_CNAME=\"sgemm_itcopy\" -DNO_AFFINITY -I$(INC_DIR) -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./$(GEMM_DIR)/gemm_tcopy_16.c -o $(BUILD_DIR)/sgemm_itcopy.o
	gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_oncopy -DASMFNAME=sgemm_oncopy_ -DNAME=sgemm_oncopy_ -DCNAME=sgemm_oncopy -DCHAR_NAME=\"sgemm_oncopy_\" -DCHAR_CNAME=\"sgemm_oncopy\" -DNO_AFFINITY -I$(INC_DIR) -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./$(GEMM_DIR)/gemm_ncopy_4.c -o $(BUILD_DIR)/sgemm_oncopy.o
	gcc -g -DMAX_STACK_ALLOC=2048 -Wall -m64 -DF_INTERFACE_GFORT -fPIC -DSMP_SERVER -DNO_WARMUP -DMAX_CPU_NUMBER=8 -DASMNAME=sgemm_kernel -DASMFNAME=sgemm_kernel_ -DNAME=sgemm_kernel_ -DCNAME=sgemm_kernel -DCHAR_NAME=\"sgemm_kernel_\" -DCHAR_CNAME=\"sgemm_kernel\" -DNO_AFFINITY -I$(INC_DIR) -UDOUBLE  -UCOMPLEX -c -UDOUBLE -UCOMPLEX ./$(GEMM_DIR)/sgemm_kernel_16x4_haswell.S -o $(BUILD_DIR)/sgemm_kernel.o
	ar  -ru $(BUILD_DIR)/libgemm.a  $(BUILD_DIR)/*.o
	ranlib  $(BUILD_DIR)/libgemm.a
	gcc -O3 -w  -o  $(BUILD_DIR)/test $(TESTS_DIR)/test.c $(SCHEDULER_DIR)/*.c -I$(UTILS_DIR) -I$(GEMM_DIR) -I$(SCHEDULER_DIR) -I$(INC_DIR) -lpthread -static -L./$(BUILD_DIR) -lgemm
	./$(BUILD_DIR)/test

build:
	mkdir $(BUILD_DIR)

clean:
	rm -rf $(BUILD_DIR)
	
