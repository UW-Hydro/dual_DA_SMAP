/* prototypes, defines and type declarations
   AWW-aug2006
*/

#include <stdio.h>
#include <stdlib.h>


/** Number of Objective functions (test comparisons) used **/
#define MAX_TEST_FUNCS    10

/** Number of Random Parameter sets Used in the solution set **/
#define MAX_SET 1000

/** Number of Parameters to Optimize **/
#define MAX_PARAM   60

/** Maximum number of Pareto ranks allowed when optimizer stops
    (1 = complete test, all points fall on Pareto solution line) **/
#define MAX_RANK 1

/** Intermediate Output Stages **/
#define PRT_GENERATION 1

/** Final Optimized Results Tolerance (Stopping Criteria) **/
#define F_TOL     1e-2

#define MAXITER   10

#define ALPHA 1.0
#define BETA 0.5
#define GAMMA 2.0
#define INVALIDMAX 1e5
#define MINVAL 1
#define MAXVAL 0
#define FALSE 0
#define TRUE  !FALSE

typedef struct {
  int   rank;
  int   pos;
  int   soln_num;
  float prob;
  float p[MAX_PARAM];
  float f[MAX_TEST_FUNCS];
} ITEM;

typedef struct {
  char name[1024];
  float max;
  float min;
} PARAM_RANGE;

typedef struct {
  /* params */
  const float *p;
  float       *f;
  int          dispatch_id;
  int         *soln_num;
  /* internal */
  char * statsfilename;
  int BADPARAM;
} DISPATCH_MODEL_STATE;

/* static int N_SET;
static int N_PARAM;
static int N_TEST_FUNCS;  */
static int SOLVE_CNT; /* now also used for concurrent dispatch -- make sure accesses this become reentrant if/when rest of code does */

typedef enum {
  amoebauninitialized, amoeba1, amoeba2, amoeba3, amoeba4, amoebadone
} AMOEBA_EXEC_STATE;

typedef struct {
  /* args */
  ITEM          test_set[MAX_PARAM], parent, spawn, r, rr, rrr;
/*  int          *extern_iter;   TODO  maybe just make this into int?   */

  /* local */
  int     FOUND;
  float   pbar[MAX_PARAM];
  float  *storef;
  int i;

  /* state */
  AMOEBA_EXEC_STATE exec_state;
  DISPATCH_MODEL_STATE * dispatch_state;
} AMOEBA_CONTEXT;



/** Function Prototypes **/
void mocom_die(const char *, ...);
void random_start_optimization(long *);
void restart_optimization(const char *);
void populate_simplex(ITEM * test_set, long *);
void amoeba(AMOEBA_CONTEXT *);
/* float **, float **, int *, int, int *, const char *, PARAM_RANGE *, const char *, FILE *); */

void solve_model(   const float *, float *, int *);
void *dispatch_model(const float *, float *, int *);
int check_model(DISPATCH_MODEL_STATE * state);
void retrieve_model(DISPATCH_MODEL_STATE *);
float ran2(long *);
int rank(ITEM *, int);
void quick(ITEM *, int);
void qs(ITEM *, int, int);

PARAM_RANGE *set_param_limits(const char *, int);

void calc_rank_probs(void);
void code_error(char *);
int  less_than(float *, float *, int);
int  less_than_or_equal(float *, float *, int);
void usage();
