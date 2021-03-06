#define SIZE    4
#define BASE_SHIFT 2
#define ZBASE_SHIFT 3

#define HALT hlt

#ifndef ALIGN_2
#define ALIGN_2 .align 4
#endif

#ifndef ALIGN_3
#define ALIGN_3 .align 8
#endif

#ifndef ALIGN_4
#define ALIGN_4 .align 16
#endif

#ifndef ALIGN_5
#define ALIGN_5 .align 32
#endif

#ifndef ALIGN_6
#define ALIGN_6 .align 64
#endif

#ifndef ffreep
#define ffreep .byte 0xdf, 0xc0 #
#endif

#define FLD	flds
#define FST	fstps
#define FSTU	fsts
#define FMUL	fmuls
#define FADD	fadds
#define MOVSD	movss
#define MULSD	mulss
#define MULPD	mulps
#define CMPEQPD	cmpeqps
#define COMISD	comiss
#define PSRLQ	psrld
#define ANDPD	andps
#define ADDPD	addps
#define ADDSD	addss
#define SUBPD	subps
#define SUBSD	subss
#define MOVQ	movd
#define MOVUPD	movups
#define XORPD	xorps

#ifndef ASSEMBLER
#define _GNU_SOURCE

// ------------------------------------------------------------------------------------

#define BLASLONG long
#define FLOAT    float
//#define GEMM_P 768
//#define GEMM_Q 384
#define GEMM_P 512
#define GEMM_Q 256
#define SGEMM_P GEMM_P
#define SGEMM_Q GEMM_Q
#define GEMM_ALIGN 16383
#define GEMM_UNROLL_N 4
#define GEMM_UNROLL_M 16
#define MAX_SUB_PTHREAD_INDEX 7
#define COMPSIZE 1


int sgemm_kernel(BLASLONG, BLASLONG, BLASLONG, float,  float  *, float  *, float  *, BLASLONG);
int sgemm_oncopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_itcopy(BLASLONG m, BLASLONG n, float *a, BLASLONG lda, float *b);
int sgemm_beta(BLASLONG, BLASLONG, BLASLONG, float, float  *, BLASLONG, float   *, BLASLONG, float  *, BLASLONG);


#define BETA_OPERATION(M_FROM, M_TO, N_FROM, N_TO, BETA, C, LDC)                 \
    sgemm_beta((M_TO) - (M_FROM),                                                \
               (N_TO - N_FROM),                                                  \
               0,                                                                \
               BETA[0],                                                          \
               NULL,                                                             \
               0,                                                                \
               NULL,                                                             \
               0,                                                                \
              (FLOAT *)(C) + ((M_FROM) + (N_FROM) * (LDC)) * COMPSIZE, LDC)


#define ICOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER)                              \
    sgemm_itcopy(M,                                                              \
                 N,                                                              \
                 (FLOAT *)(A) + ((Y) + (X) * (LDA)) * COMPSIZE, LDA, BUFFER);


#define OCOPY_OPERATION(M, N, A, LDA, X, Y, BUFFER)                              \
    sgemm_oncopy(M,                                                              \
                 N,                                                              \
                 (FLOAT *)(A) + ((X) + (Y) * (LDA)) * COMPSIZE, LDA, BUFFER);


#define KERNEL_OPERATION(M, N, K, ALPHA, SA, SB, C, LDC, X, Y)                   \
        sgemm_kernel(M,                                                          \
                     N,                                                          \
                     K,                                                          \
                     ALPHA[0],                                                   \
                     SA,                                                         \
                     SB,                                                         \
                     (FLOAT *)(C) + ((X) + (Y) * LDC) * COMPSIZE, LDC)


#define WMB

// ------------------------------------------------------------------------------------


#ifdef XDOUBLE
#define GET_IMAGE(res)  __asm__ __volatile__("fstpt %0" : "=m"(res) : : "memory")
#elif defined(DOUBLE)
#define GET_IMAGE(res)  __asm__ __volatile__("movsd %%xmm1, %0" : "=m"(res) : : "memory")
#else
#define GET_IMAGE(res)  __asm__ __volatile__("movss %%xmm1, %0" : "=m"(res) : : "memory")
#endif

#ifndef PAGESIZE
#define PAGESIZE      ( 4 << 10)
#endif
#define HUGE_PAGESIZE ( 2 << 20)

#define BUFFER_SIZE   ( 2 << 20)

#endif

#ifdef ASSEMBLER

#if defined(PILEDRIVER) || defined(BULLDOZER) || defined(STEAMROLLER) || defined(EXCAVATOR)
//Enable some optimazation for barcelona.
#define BARCELONA_OPTIMIZATION
#endif

#if defined(HAVE_3DNOW)
#define EMMS	femms
#elif defined(HAVE_MMX)
#define EMMS	emms
#endif

#ifndef EMMS
#define EMMS
#endif

#define BRANCH		.byte 0x3e
#define NOBRANCH	.byte 0x2e
#define PADDING		.byte 0x66

#ifdef OS_WINDOWS
#define ARG1	%rcx
#define ARG2	%rdx
#define ARG3	%r8
#define ARG4	%r9
#else
#define ARG1	%rdi
#define ARG2	%rsi
#define ARG3	%rdx
#define ARG4	%rcx
#define ARG5	%r8
#define ARG6	%r9
#endif

#ifndef COMPLEX
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  QLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  DLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  SLOCAL_BUFFER_SIZE
#endif
#else
#ifdef XDOUBLE
#define LOCAL_BUFFER_SIZE  XLOCAL_BUFFER_SIZE
#elif defined DOUBLE
#define LOCAL_BUFFER_SIZE  ZLOCAL_BUFFER_SIZE
#else
#define LOCAL_BUFFER_SIZE  CLOCAL_BUFFER_SIZE
#endif
#endif

#if defined(OS_WINDOWS)
#if   LOCAL_BUFFER_SIZE > 16384
#define STACK_TOUCHING \
	movl	$0,  4096 * 4(%rsp);\
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 12288
#define STACK_TOUCHING \
	movl	$0,  4096 * 3(%rsp);\
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 8192
#define STACK_TOUCHING \
	movl	$0,  4096 * 2(%rsp);\
	movl	$0,  4096 * 1(%rsp);
#elif LOCAL_BUFFER_SIZE > 4096
#define STACK_TOUCHING \
	movl	$0,  4096 * 1(%rsp);
#else
#define STACK_TOUCHING
#endif
#else
#define STACK_TOUCHING
#endif

#if defined(CORE2)
#define movapd	movaps
#define andpd	andps
#define movlpd	movlps
#define movhpd	movhps
#endif

#ifndef F_INTERFACE
#define REALNAME ASMNAME
#else
#define REALNAME ASMFNAME
#endif


// ------------------------------------------------------------------------------------

#if defined(OS_LINUX) || defined(OS_FREEBSD) || defined(OS_NETBSD) || defined(__ELF__) || defined(C_PGI)
#define PROLOGUE \
	.text; \
	.align 512; \
	.globl REALNAME ;\
       .type REALNAME, @function; \
REALNAME:

#ifdef PROFILE
#define PROFCODE call *mcount@GOTPCREL(%rip)
#else
#define PROFCODE
#endif

#define EPILOGUE \
        .size	 REALNAME, .-REALNAME; \
        .section .note.GNU-stack,"",@progbits

#endif

#endif 

