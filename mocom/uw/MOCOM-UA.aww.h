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
#define MAX_PARAM   10

/** Maximum number of Pareto ranks allowed when optimizer stops 
    (1 = complete test, all points fall on Pareto solution line) **/
#define MAX_RANK 1  /* doesn't allow testing ranks higher than 1? AWW */

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
  float y;
  float f[MAX_TEST_FUNCS];
} ITEM;

typedef struct {
  char name[10];
  float max;
  float min;
} PARAM_RANGE;

static int N_SET;
static int N_PARAM;
static int N_TEST_FUNCS;
static int SOLVE_CNT;

/** Function Prototypes **/
void random_start_optimization(FILE *, ITEM *set, PARAM_RANGE *, int *,
			       int, int, int, int *, float **, long *, char *, 
			       char *, char *, int);
void restart_optimization(char *, FILE *fopti, ITEM *set, PARAM_RANGE *,
			  int *, int, int, int *, float **, int);
void         nrerror(char *);
int         amoeba(float **, float *, float **, int *, int, float, 
		   int *, char *, PARAM_RANGE *, char *, char *, int, 
		    FILE *);
float        solve_model(float *, float *, char *, PARAM_RANGE *, char *, 
			 FILE *, char *, int);
float        ran2(long *);
int          rank(ITEM *);
void         quick(ITEM *, int);
void         qs(ITEM *, int, int);
PARAM_RANGE *set_param_limits(char *, int);
float       *calc_rank_probs(ITEM *, int, int*);
void         code_error(char *);
int          less_than(float *, float *, int);
int          less_than_or_equal(float *, float *, int);
void         usage();
