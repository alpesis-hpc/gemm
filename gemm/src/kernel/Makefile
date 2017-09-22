CC = gcc
PFLAGS     += $(COMMON_OPT) $(CCOMMON_OPT) -I$(TOPDIR) -DPROFILE $(COMMON_PROF)
COMMON_OPT = -O2
KDIR = ./
TSUFFIX =
PSUFFIX = pobj

ifdef TARGET_CORE
TARGET = $(TARGET_CORE)
endif


ifdef TARGET_CORE
override CFLAGS += -DBUILD_KERNEL -DTABLE_NAME=gotoblas_$(TARGET_CORE)
BUILD_KERNEL = 1
KDIR =
TSUFFIX = _$(TARGET_CORE)
else
TARGET_CORE = $(CORE)
KDIR =
TSUFFIX =
endif

-include x86_64/KERNEL.$(TARGET_CORE)

include x86_64/KERNEL


include Makefile.L3


HPLOBJS = \
	dgemm_kernel.$(SUFFIX) \
	$(DGEMMINCOPYOBJ) $(DGEMMITCOPYOBJ) \
	$(DGEMMONCOPYOBJ) $(DGEMMOTCOPYOBJ) \
	dtrsm_kernel_LN.$(SUFFIX) dtrsm_kernel_LT.$(SUFFIX) \
	dtrsm_kernel_RN.$(SUFFIX) dtrsm_kernel_RT.$(SUFFIX) \
	daxpy_k.$(SUFFIX) dcopy_k.$(SUFFIX) ddot_k.$(SUFFIX) \
	dger_k.$(SUFFIX)  dscal_k.$(SUFFIX) idamax_k.$(SUFFIX) \
	dgemv_n.$(SUFFIX) dgemv_t.$(SUFFIX) dgemm_beta.$(SUFFIX) \
	dtrsm_iunucopy.$(SUFFIX) dtrsm_iunncopy.$(SUFFIX) \
	dtrsm_ilnucopy.$(SUFFIX) dtrsm_ilnncopy.$(SUFFIX) \
	dtrsm_iutucopy.$(SUFFIX) dtrsm_iutncopy.$(SUFFIX) \
	dtrsm_iltucopy.$(SUFFIX) dtrsm_iltncopy.$(SUFFIX) \
	dtrsm_ounucopy.$(SUFFIX) dtrsm_ounncopy.$(SUFFIX) \
	dtrsm_olnucopy.$(SUFFIX) dtrsm_olnncopy.$(SUFFIX) \
	dtrsm_outucopy.$(SUFFIX) dtrsm_outncopy.$(SUFFIX) \
	dtrsm_oltucopy.$(SUFFIX) dtrsm_oltncopy.$(SUFFIX)

COMMONOBJS	+= lsame.$(SUFFIX) scabs1.$(SUFFIX) dcabs1.$(SUFFIX)

ifeq ($(DYNAMIC_ARCH), 1)
SBLASOBJS	+= setparam$(TSUFFIX).$(SUFFIX)
CCOMMON_OPT	+= -DTS=$(TSUFFIX)
endif

KERNEL_INTERFACE = ../common_level1.h ../common_level2.h ../common_level3.h
ifneq ($(NO_LAPACK), 1)
KERNEL_INTERFACE += ../common_lapack.h
endif

ifeq ($(ARCH), x86)
COMMONOBJS	+= cpuid.$(SUFFIX)
endif

ifdef EXPRECISION
COMMONOBJS	+= qconjg.$(SUFFIX) qcabs1.$(SUFFIX)
endif

ifdef QUAD_PRECISION
COMMONOBJS	+= qconjg.$(SUFFIX) qcabs1.$(SUFFIX)
endif

all : libs

libs    :: $(BLASOBJS) $(COMMONOBJS)
	$(AR) $(ARFLAGS) -ru $(TOPDIR)/$(LIBNAME) $^



scabs1.$(SUFFIX): $(KERNELDIR)/$(SCABS_KERNEL)
	$(CC) -c $(CFLAGS) -DCOMPLEX -UDOUBLE -DF_INTERFACE $< -o $(@F)

dcabs1.$(SUFFIX): $(KERNELDIR)/$(DCABS_KERNEL)
	$(CC) -c $(CFLAGS) -DCOMPLEX -DDOUBLE -DF_INTERFACE $< -o $(@F)

qcabs1.$(SUFFIX): $(KERNELDIR)/$(QCABS_KERNEL)
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE -DF_INTERFACE $< -o $(@F)

qconjg.$(SUFFIX): $(KERNELDIR)/qconjg.S
	$(CC) -c $(CFLAGS) -DCOMPLEX -DXDOUBLE -DF_INTERFACE $< -o $(@F)

lsame.$(SUFFIX): $(KERNELDIR)/$(LSAME_KERNEL)
	$(CC) -c $(CFLAGS) -DF_INTERFACE $< -o $(@F)

setparam$(TSUFFIX).$(SUFFIX): setparam$(TSUFFIX).c kernel$(TSUFFIX).h
	$(CC) -c $(CFLAGS) $< -o $@

setparam$(TSUFFIX).c : setparam-ref.c
	sed 's/TS/$(TSUFFIX)/g' $< > $(@F)

kernel$(TSUFFIX).h : $(KERNEL_INTERFACE)
	sed 's/\ *(/$(TSUFFIX)(/g' $^ > $(@F)


cpuid.$(SUFFIX): $(KERNELDIR)/cpuid.S
	$(CC) -c $(CFLAGS) $< -o $(@F)


#ifdef DYNAMIC_ARCH
clean ::
	@rm -f setparam_*.c kernel_*.h setparam.h kernel.h

#endif

