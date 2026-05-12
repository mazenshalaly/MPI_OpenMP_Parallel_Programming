/*
 * ============================================================
 *  Parallel System: MPI + OpenMP Integration
 * ============================================================
 *  Process Layout (10 processes total):
 *    Rank 0  : Master  — distributes tasks, collects results
 *    Rank 1  : Integer worker  — Factorial (OpenMP parallel)
 *    Rank 2  : String  worker  — Palindrome + reverse + vowels (OpenMP)
 *    Rank 3  : File    worker  — split even/odd lines (OpenMP)
 *    Rank 4  : Matrix  worker  — coordinates matrix ops
 *    Ranks 5-9: Matrix sub-workers (used by rank 4 via MPI)
 * ============================================================
 */

#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <stdint.h>

/* ── tuneable constants ─────────────────────────────────── */
#define MATRIX_N        50
#define MAX_STR         512
#define MAX_LINE        256
#define MAX_FILE_LINES  100
#define NUM_MATRIX_WORKERS 5   /* ranks 5-9 */

/* ── MPI tags ───────────────────────────────────────────── */
#define TAG_INT_DATA    10
#define TAG_INT_RESULT  11
#define TAG_STR_DATA    20
#define TAG_STR_RESULT  21
#define TAG_FILE_DATA   30
#define TAG_FILE_RESULT 31
#define TAG_MAT_DATA    40
#define TAG_MAT_RESULT  41
#define TAG_MAT_SUB     50   /* inter-subworker tag */

/* ── tiny helpers ───────────────────────────────────────── */
static void print_separator(void) {
    printf("──────────────────────────────────────────────────────\n");
}

/* ═══════════════════════════════════════════════════════════
 *  RANK 1 — Integer worker: Factorial with OpenMP reduction
 * ═══════════════════════════════════════════════════════════ */
static void worker_integer(int rank)
{
    int n;
    MPI_Recv(&n, 1, MPI_INT, 0, TAG_INT_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Parallel factorial using OpenMP reduction */
    unsigned long long result = 1ULL;

    /* Split multiplication across threads */
    #pragma omp parallel reduction(*:result)
    {
        int tid   = omp_get_thread_num();
        int nthrd = omp_get_num_threads();
        for (int i = tid + 1; i <= n; i += nthrd)
            result *= (unsigned long long)i;
    }

    /* Also compute power of 4 */
    unsigned long long pow4 = 1ULL;
    #pragma omp parallel reduction(*:pow4)
    {
        int tid   = omp_get_thread_num();
        int nthrd = omp_get_num_threads();
        for (int i = tid; i < n; i += nthrd)
            pow4 *= 4ULL;
    }

    /* Pack both results into an array and send */
    unsigned long long results[2] = { result, pow4 };
    MPI_Send(results, 2, MPI_UNSIGNED_LONG_LONG, 0, TAG_INT_RESULT, MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════
 *  RANK 2 — String worker: palindrome / reverse / vowels
 * ═══════════════════════════════════════════════════════════ */
static void worker_string(int rank)
{
    char str[MAX_STR];
    MPI_Recv(str, MAX_STR, MPI_CHAR, 0, TAG_STR_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    int len = (int)strlen(str);

    /* 1) Palindrome check — parallel comparison */
    int is_palindrome = 1;
    #pragma omp parallel for reduction(&&:is_palindrome)
    for (int i = 0; i < len / 2; i++) {
        if (tolower((unsigned char)str[i]) != tolower((unsigned char)str[len - 1 - i]))
            is_palindrome = 0;
    }

    /* 2) Count vowels — parallel reduction */
    int vowel_count = 0;
    #pragma omp parallel for reduction(+:vowel_count)
    for (int i = 0; i < len; i++) {
        char c = tolower((unsigned char)str[i]);
        if (c=='a'||c=='e'||c=='i'||c=='o'||c=='u')
            vowel_count++;
    }

    /* 3) Reverse string — parallel swap */
    char reversed[MAX_STR];
    memcpy(reversed, str, len + 1);
    #pragma omp parallel for
    for (int i = 0; i < len / 2; i++) {
        char tmp         = reversed[i];
        reversed[i]      = reversed[len - 1 - i];
        reversed[len-1-i]= tmp;
    }

    /* Pack results: palindrome(int) + vowels(int) + reversed string */
    char result_buf[MAX_STR + 64];
    snprintf(result_buf, sizeof(result_buf),
             "PALINDROME:%d|VOWELS:%d|REVERSED:%s",
             is_palindrome, vowel_count, reversed);

    MPI_Send(result_buf, (int)strlen(result_buf)+1, MPI_CHAR, 0,
             TAG_STR_RESULT, MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════
 *  RANK 3 — File worker: split even/odd lines with OpenMP
 * ═══════════════════════════════════════════════════════════ */
static void worker_file(int rank)
{
    /* Receive the filename */
    char filename[MAX_STR];
    MPI_Recv(filename, MAX_STR, MPI_CHAR, 0, TAG_FILE_DATA, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Read all lines first */
    FILE *fin = fopen(filename, "r");
    char lines[MAX_FILE_LINES][MAX_LINE];
    int  total = 0;

    if (fin) {
        while (total < MAX_FILE_LINES && fgets(lines[total], MAX_LINE, fin))
            total++;
        fclose(fin);
    }

    /* Separate even/odd indices in parallel */
    char even_lines[MAX_FILE_LINES][MAX_LINE];
    char odd_lines [MAX_FILE_LINES][MAX_LINE];
    int  even_cnt = 0, odd_cnt = 0;

    /* Serial collection into separate arrays (parallel read of lines array) */
    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i < total; i++) {
                if (i % 2 == 0)
                    strncpy(even_lines[even_cnt++], lines[i], MAX_LINE);
                else
                    strncpy(odd_lines [odd_cnt++ ], lines[i], MAX_LINE);
            }
        }
    }

    /* Write output files */
    FILE *fe = fopen("even_lines.txt", "w");
    FILE *fo = fopen("odd_lines.txt",  "w");

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if (fe) {
                for (int i = 0; i < even_cnt; i++)
                    fputs(even_lines[i], fe);
                fclose(fe);
            }
        }
        #pragma omp section
        {
            if (fo) {
                for (int i = 0; i < odd_cnt; i++)
                    fputs(odd_lines[i], fo);
                fclose(fo);
            }
        }
    }

    char result_buf[256];
    snprintf(result_buf, sizeof(result_buf),
             "TOTAL_LINES:%d|EVEN_LINES:%d|ODD_LINES:%d",
             total, even_cnt, odd_cnt);

    MPI_Send(result_buf, (int)strlen(result_buf)+1, MPI_CHAR, 0,
             TAG_FILE_RESULT, MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════
 *  RANK 4 — Matrix coordinator + sub-workers (ranks 5-9)
 *
 *  Operations performed (all 50×50):
 *    1. Matrix + scalar (distributed across sub-workers)
 *    2. Matrix transpose (OpenMP local)
 *    3. Element-wise multiply with another matrix (parallel)
 * ═══════════════════════════════════════════════════════════ */

#define MAT_SIZE (MATRIX_N * MATRIX_N)

static void worker_matrix_sub(int rank)
{
    /* Each sub-worker handles a horizontal stripe of the matrix.
       We receive: [rows_in_stripe * MATRIX_N] floats + offset info */
    float stripe[MATRIX_N * MATRIX_N]; /* over-allocated for simplicity */
    MPI_Status st;
    MPI_Recv(stripe, MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_SUB,
             MPI_COMM_WORLD, &st);

    int count;
    MPI_Get_count(&st, MPI_FLOAT, &count);
    int rows = count / MATRIX_N;

    /* Operation: add 1.0 to every element (parallelised with OpenMP) */
    #pragma omp parallel for
    for (int i = 0; i < rows * MATRIX_N; i++)
        stripe[i] += 1.0f;

    MPI_Send(stripe, count, MPI_FLOAT, 4, TAG_MAT_SUB, MPI_COMM_WORLD);
}

static void worker_matrix(int rank)
{
    /* Receive the full 50×50 matrix A and matrix B from master */
    float A[MAT_SIZE], B[MAT_SIZE];
    MPI_Recv(A, MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_DATA,     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(B, MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_DATA + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* ── Operation 1: Distribute stripes to sub-workers for A+1 ── */
    int rows_per = MATRIX_N / NUM_MATRIX_WORKERS;      /* 10 rows each */
    int leftover  = MATRIX_N % NUM_MATRIX_WORKERS;

    float A_plus1[MAT_SIZE];

    for (int w = 0; w < NUM_MATRIX_WORKERS; w++) {
        int start = w * rows_per * MATRIX_N;
        int rcount = rows_per + (w == NUM_MATRIX_WORKERS-1 ? leftover : 0);
        MPI_Send(A + start, rcount * MATRIX_N, MPI_FLOAT,
                 5 + w, TAG_MAT_SUB, MPI_COMM_WORLD);
    }
    for (int w = 0; w < NUM_MATRIX_WORKERS; w++) {
        int start  = w * rows_per * MATRIX_N;
        int rcount = rows_per + (w == NUM_MATRIX_WORKERS-1 ? leftover : 0);
        MPI_Recv(A_plus1 + start, rcount * MATRIX_N, MPI_FLOAT,
                 5 + w, TAG_MAT_SUB, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    /* ── Operation 2: Transpose A (OpenMP) ── */
    float A_T[MAT_SIZE];
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < MATRIX_N; i++)
        for (int j = 0; j < MATRIX_N; j++)
            A_T[j * MATRIX_N + i] = A[i * MATRIX_N + j];

    /* ── Operation 3: Element-wise A.*B (OpenMP) ── */
    float AB_elem[MAT_SIZE];
    #pragma omp parallel for
    for (int k = 0; k < MAT_SIZE; k++)
        AB_elem[k] = A[k] * B[k];

    /* ── Operation 4: Matrix addition A+B (OpenMP) ── */
    float AB_sum[MAT_SIZE];
    #pragma omp parallel for
    for (int k = 0; k < MAT_SIZE; k++)
        AB_sum[k] = A[k] + B[k];

    /* Send all four result matrices back to master */
    MPI_Send(A_plus1,  MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_RESULT,     MPI_COMM_WORLD);
    MPI_Send(A_T,      MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_RESULT + 1, MPI_COMM_WORLD);
    MPI_Send(AB_elem,  MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_RESULT + 2, MPI_COMM_WORLD);
    MPI_Send(AB_sum,   MAT_SIZE, MPI_FLOAT, 0, TAG_MAT_RESULT + 3, MPI_COMM_WORLD);
}

/* ═══════════════════════════════════════════════════════════
 *  Helper: pretty-print a corner of a matrix
 * ═══════════════════════════════════════════════════════════ */
static void print_matrix_corner(const char *label, const float *M, int n, int corner)
{
    printf("  %s (top-%dx%d corner):\n", label, corner, corner);
    for (int i = 0; i < corner; i++) {
        printf("    [ ");
        for (int j = 0; j < corner; j++)
            printf("%6.1f ", M[i * n + j]);
        printf("... ]\n");
    }
    printf("    ...\n");
}

/* ═══════════════════════════════════════════════════════════
 *  RANK 0 — Master process
 * ═══════════════════════════════════════════════════════════ */
static void master(void)
{
    printf("\n");
    print_separator();
    printf("  PARALLEL SYSTEM — MPI + OpenMP  (Master: rank 0)\n");
    print_separator();
    printf("  Processes : 10  (1 master + 4 workers + 5 matrix sub-workers)\n");
    printf("  Matrix    : %dx%d\n", MATRIX_N, MATRIX_N);
    printf("  Threads   : %d per MPI process\n", omp_get_max_threads());
    print_separator();
    printf("\n");

    /* ── Task 1: Integer data → rank 1 ──────────────────── */
    int n = 12;
    printf("[Master] Sending integer n=%d to rank 1 (Factorial + Power-of-4)\n", n);
    MPI_Send(&n, 1, MPI_INT, 1, TAG_INT_DATA, MPI_COMM_WORLD);

    /* ── Task 2: String data → rank 2 ───────────────────── */
    const char *input_str = "racecar";
    printf("[Master] Sending string \"%s\" to rank 2 (palindrome/vowels/reverse)\n", input_str);
    MPI_Send(input_str, (int)strlen(input_str)+1, MPI_CHAR, 2,
             TAG_STR_DATA, MPI_COMM_WORLD);

    /* ── Task 3: File path → rank 3 ─────────────────────── */
    const char *filepath = "input.txt";
    printf("[Master] Sending file \"%s\" to rank 3 (even/odd line split)\n", filepath);
    MPI_Send(filepath, (int)strlen(filepath)+1, MPI_CHAR, 3,
             TAG_FILE_DATA, MPI_COMM_WORLD);

    /* ── Task 4: Two 50×50 matrices → rank 4 ───────────── */
    printf("[Master] Sending two 50x50 matrices to rank 4 (matrix ops)\n");
    float *A = (float*)malloc(MAT_SIZE * sizeof(float));
    float *B = (float*)malloc(MAT_SIZE * sizeof(float));

    /* Fill A with sequential values, B with constant 2.0 */
    #pragma omp parallel for
    for (int i = 0; i < MAT_SIZE; i++) {
        A[i] = (float)(i % 10) + 1.0f;   /* 1..10 cycling */
        B[i] = 2.0f;
    }

    MPI_Send(A, MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_DATA,     MPI_COMM_WORLD);
    MPI_Send(B, MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_DATA + 1, MPI_COMM_WORLD);

    /* ═══ Collect results ═══════════════════════════════════ */
    printf("\n[Master] Waiting for results...\n\n");

    /* Result 1: integer */
    unsigned long long int_results[2];
    MPI_Recv(int_results, 2, MPI_UNSIGNED_LONG_LONG, 1,
             TAG_INT_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Result 2: string */
    char str_result[MAX_STR + 64];
    MPI_Recv(str_result, sizeof(str_result), MPI_CHAR, 2,
             TAG_STR_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Result 3: file */
    char file_result[256];
    MPI_Recv(file_result, sizeof(file_result), MPI_CHAR, 3,
             TAG_FILE_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Result 4: matrices (4 operations) */
    float *A_plus1  = (float*)malloc(MAT_SIZE * sizeof(float));
    float *A_T      = (float*)malloc(MAT_SIZE * sizeof(float));
    float *AB_elem  = (float*)malloc(MAT_SIZE * sizeof(float));
    float *AB_sum   = (float*)malloc(MAT_SIZE * sizeof(float));

    MPI_Recv(A_plus1, MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_RESULT,     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(A_T,     MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_RESULT + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(AB_elem, MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_RESULT + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(AB_sum,  MAT_SIZE, MPI_FLOAT, 4, TAG_MAT_RESULT + 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* ═══ Display all results ═══════════════════════════════ */
    print_separator();
    printf("  RESULTS\n");
    print_separator();

    /* Integer results */
    printf("\n╔══ RANK 1 — Integer Operations (n=%d) ══╗\n", n);
    printf("  Factorial(%d)  = %llu\n", n, int_results[0]);
    printf("  4^%d           = %llu\n", n, int_results[1]);

    /* String results */
    printf("\n╔══ RANK 2 — String Operations (\"%s\") ══╗\n", input_str);
    /* Parse the packed result */
    int is_pal = 0, vowels = 0;
    char reversed[MAX_STR] = "";
    sscanf(str_result, "PALINDROME:%d|VOWELS:%d|REVERSED:%511s",
           &is_pal, &vowels, reversed);
    printf("  Is Palindrome : %s\n", is_pal ? "YES" : "NO");
    printf("  Vowel Count   : %d\n", vowels);
    printf("  Reversed      : \"%s\"\n", reversed);

    /* File results */
    printf("\n╔══ RANK 3 — File Split Operations ══╗\n");
    int total_l=0, even_l=0, odd_l=0;
    sscanf(file_result, "TOTAL_LINES:%d|EVEN_LINES:%d|ODD_LINES:%d",
           &total_l, &even_l, &odd_l);
    printf("  Input file    : %s\n", filepath);
    printf("  Total lines   : %d\n", total_l);
    printf("  Even-index    : %d lines → even_lines.txt\n", even_l);
    printf("  Odd-index     : %d lines → odd_lines.txt\n",  odd_l);

    /* Matrix results */
    printf("\n╔══ RANK 4+5-9 — Matrix Operations (50×50) ══╗\n");
    print_matrix_corner("A (original)",    A,       MATRIX_N, 4);
    print_matrix_corner("B (constant 2)",  B,       MATRIX_N, 4);
    print_matrix_corner("A+1 (sub-workers distribute)", A_plus1, MATRIX_N, 4);
    print_matrix_corner("A^T (transpose)", A_T,     MATRIX_N, 4);
    print_matrix_corner("A.*B (elem-wise)", AB_elem, MATRIX_N, 4);
    print_matrix_corner("A+B (addition)",  AB_sum,  MATRIX_N, 4);

    /* Verify spot-check */
    printf("\n  Verification spot-checks:\n");
    printf("    A[0][0]=%.1f  →  A+1[0][0]=%.1f  (expect %.1f) %s\n",
           A[0], A_plus1[0], A[0]+1.0f,
           (A_plus1[0] == A[0]+1.0f) ? "✓" : "✗");
    printf("    A[1][2]=%.1f  →  A^T[2][1]=%.1f  (expect %.1f) %s\n",
           A[1*MATRIX_N+2], A_T[2*MATRIX_N+1], A[1*MATRIX_N+2],
           (A_T[2*MATRIX_N+1] == A[1*MATRIX_N+2]) ? "✓" : "✗");
    printf("    A[0][0]*B[0][0]=%.1f*%.1f  →  elem[0][0]=%.1f %s\n",
           A[0], B[0], AB_elem[0],
           (AB_elem[0] == A[0]*B[0]) ? "✓" : "✗");
    printf("    A[0][0]+B[0][0]=%.1f+%.1f  →  sum[0][0]=%.1f %s\n",
           A[0], B[0], AB_sum[0],
           (AB_sum[0] == A[0]+B[0]) ? "✓" : "✗");

    print_separator();
    printf("  All tasks completed successfully.\n");
    print_separator();
    printf("\n");

    free(A); free(B);
    free(A_plus1); free(A_T); free(AB_elem); free(AB_sum);
}

/* ═══════════════════════════════════════════════════════════
 *  main
 * ═══════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 10) {
        if (rank == 0)
            fprintf(stderr,
                "ERROR: Need at least 10 processes. Got %d.\n"
                "Run with: mpirun -np 10 ./parallel_system\n", size);
        MPI_Finalize();
        return 1;
    }

    /* Set OpenMP threads per process */
    omp_set_num_threads(4);

    switch (rank) {
        case 0: master();              break;
        case 1: worker_integer(rank);  break;
        case 2: worker_string(rank);   break;
        case 3: worker_file(rank);     break;
        case 4: worker_matrix(rank);   break;
        default:
            if (rank >= 5 && rank <= 9)
                worker_matrix_sub(rank);
            break;
    }

    MPI_Finalize();
    return 0;
}
