#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "MOCOM-UA.aww.h"

int main(int argc,char *argv[]) {
/**********************************************************************
  MOCOM-UA.c             Keith Cherkauer            November 18, 1999

  This program optimizes the provided model script using the 
  multi-objective global optimization method introduced in:
  Yapo, P. O., H. V. Gupta, and S. Sorooshian, "Multi-objective
  global optimization for hydrologic models," J. Hydrol., 204,
  1998, pp. 83-97.

  The program requires:
  (1) the name of a script which will change the parameters of the 
  model, run it, route the results, and compute an R^2 value (times 
  -1 so that an R^2 of -1 is a perfect fit) which is then returned 
  to this program and used to find optimized parameters.

  (2) a prefix unique to this run of the program with which to create
  a temporary file.

  (3) the name of a log file to which it can write the results of
  all optimization attempts.

  Modifications:
  09252000 Modified to let the user define an initial population larger
    than the one actually used to find the Pareto line.  The program
    first determines the results for NUM_RAND random starts, ranks and
    sorts those starts and then optimizes the NUM_SET best values
    from the random start.                                        KAC
  01102001 Modified so that the optimizer can be restarted if provided
    with a file containing a list of parameters, their resulting test
    values and the storage directory if used in the previous 
    simulations.                                                  KAC
  05152003 Modified to store every 100 solution sets in a new 
    subdirectory (sets 0-99 are in SAVE_0, sets 100-199 are in SAVE_100).
    The optimizer also removes all sets that were not saved in an
    output generation list, which should reduce the number of saved
    sets.                                                         KAC
  20060801 Modified in several ways:  added additional check to main interation
    loop to avoid endless loop in one circumstance that arises mostly with
    single optimization.  Also added documentation, particularly in amoeba and a
    few subroutines; also moved Usage statement to function; also moved lots of
    declarations at top into an include file to avoid code clutter; and cleaned
    up i/o a bit.  Keith has since added one restart option that is not in this
    code.  -AWW

**********************************************************************/

  extern int N_SET;
  extern int N_PARAM;
  extern int N_TEST_FUNCS;
  extern int SOLVE_CNT;

  FILE         *fopti;
  float       **p, *y;
  int           RESTART;
  int           i, j, k, l, try, param, iter;
  char          runstr[512];
  char          labelstr[512];
  char          param_lim_file[512];
  char          storedir[1024];
  char          restart_file[1024];
  int           storefiles;
  PARAM_RANGE  *param_lim;
  float        *prob, tmp_prob;
  float       **f;
  float        *Dcheck;
  int          *soln_num;
  int           Rmax;
  int           N_Rmax;
  int           N_RAND;
  int           NEW_SET;
  int           FOUND_BETTER;
  int           minsum;
  int           mincheck;
  int           test;
  int           cycle;  /* AWW-test param*/
  time_t        currtime, tmptime;
  long         *ran2seed;
  ITEM          test_set[MAX_PARAM+1];
  ITEM          set[MAX_SET];
  char          cmdstr[1024];

  if ( argc < 9 || argc > 10 ) {
    usage();
  }
  
  if ( argv[1][0] >= 48 && argv[1][0] <= 57 ) {
    /* random start selected */
    RESTART = FALSE;
    N_RAND = atoi(argv[1]);  /* number of random parameter sets to generate */
    if(N_RAND > MAX_SET) {
      fprintf(stderr,"ERROR: Requested population size %i, is larger than MAX_SET (%i).\nReduce the population or recompile the optimizer.\n",N_RAND,MAX_SET);
      exit(0);
    }
  }
  else {
    /* restart selected */
    RESTART = TRUE;
    strcpy(restart_file,argv[1]);
  }
  N_SET = atoi(argv[2]);
  if(N_SET > MAX_SET) {
    fprintf(stderr,"ERROR: Requested population size %i, is larger than MAX_SET (%i).\nReduce the population or recompile the optimizer.\n",N_SET,MAX_SET);
    exit(0);
  }
  N_PARAM = atoi(argv[3]);
  if(N_PARAM > MAX_PARAM) {
    fprintf(stderr,"ERROR: Requested number of parameters %i, is larger than MAX_PARAM (%i).\nReduce the number of parameters or recompile the optimizer.\n",N_PARAM,MAX_PARAM);
    exit(0);
  }
  N_TEST_FUNCS = atoi(argv[4]);
  if(N_TEST_FUNCS > MAX_TEST_FUNCS) {
    fprintf(stderr,"ERROR: Requested number of test functions %i, is greater than MAX_TEST_FUNCS (%i).\nReduce the number of test statistics or recompile the optimizer.\n",N_TEST_FUNCS,MAX_TEST_FUNCS);
    exit(0);
  }
  strcpy(runstr,argv[5]);
  strcpy(labelstr,argv[6]);
  if((fopti=fopen(argv[7],"w"))==NULL) {
    fprintf(stderr,"ERROR: Unable to open optimization log file %s.\n",
	    argv[7]);
    exit(0);
  }
  strcpy(param_lim_file,argv[8]);
  if ( argc == 10 ) {
    strcpy(storedir,argv[9]);
    storefiles = TRUE;
  }
  else storefiles = FALSE;

  ///////////////

    fprintf(stderr,"inputs are; %s %s %s %s %s %s %s %s %s\n",argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8], argv[9]);

  /////////////////////////////////

  /** Seed the random number generator **/
  tmptime     = time(&currtime);
  tmptime    *= -1;
  ran2seed    = &tmptime;
  ran2(ran2seed);
  ran2seed[0] = 1;

  /** Initialize variables and arrays **/
  SOLVE_CNT = 0;
  try       = 0;

  p        = (float**)calloc(N_PARAM+1,sizeof(float*));
  y        = (float *)calloc(N_PARAM+1,sizeof(float));
  Dcheck   = (float *)calloc(N_PARAM+1,sizeof(float));
  soln_num = (int *)calloc(N_PARAM+1,sizeof(int));
  f        = (float**)calloc(N_PARAM+1,sizeof(float*));
  for(i=0;i<N_PARAM+1;i++) 
    p[i] = (float*)calloc(N_PARAM,sizeof(float));
  for(i=0;i<N_PARAM+1;i++) 
    f[i] = (float *)calloc(N_TEST_FUNCS,sizeof(float));

  /** set parameter limits **/
  param_lim = set_param_limits(param_lim_file,N_PARAM);


  /** Initialize first generation **/
  printf("starting initial set generation\n"); 
  if ( RESTART ) restart_optimization(restart_file, fopti, set, param_lim,
				      &N_Rmax, N_SET, N_TEST_FUNCS, &Rmax,
				      &prob, storefiles);
  else random_start_optimization(fopti, set, param_lim, &N_Rmax, N_RAND,
				 N_SET, N_TEST_FUNCS, &Rmax, &prob, ran2seed, 
				 runstr, labelstr, storedir, storefiles);
  /* 
     if using random_start, this generates N_RAND parameters sets and runs model
     for each, then checks the stats and reduces them to the best N_SET sets.  
  */

  printf("ending initial set generation -- starting tests\n\n"); 
  /*  while ( Rmax > MAX_RANK ) {    AWW: replaced w/ following */
  while ( Rmax > MAX_RANK && 
          (N_SET - N_Rmax) > N_PARAM ) {

    if(try % PRT_GENERATION == 0) {

      /** Print current generation for external monitoring **/

      fprintf(fopti,"\nCurrent generation for try %i:\n", try);
      fprintf(fopti,"\n==========\n");
      /*header row*/
      for ( j = 0; j < N_PARAM; j++ ) 
	fprintf(fopti,"\t%s",param_lim[j].name);
      for ( j = 0; j < N_TEST_FUNCS; j++ ) 
	fprintf(fopti,"\ttest%d",j);
      if ( storefiles ) {
	fprintf(fopti," \trank\tsoln_num\n", set[i].rank, set[i].soln_num);
      }
      fprintf(fopti,"\n");
      /*run specs*/
      for ( i = 0; i < N_SET; i++) {
	fprintf(fopti,"%i:",i);
	for ( j = 0; j < N_PARAM; j++ ) 
	  fprintf(fopti,"\t%.5g",set[i].p[j]);
	fprintf(fopti,"\t= (");
	for ( j = 0; j < N_TEST_FUNCS; j++ ) {
	  fprintf(fopti,"\t%.5g",set[i].f[j]);
	}
	if ( storefiles ) {
	  fprintf(fopti," )\t%i\t%i\n", set[i].rank, set[i].soln_num);
	  // identify current set for saving
	  sprintf( cmdstr, "if [ -d %s/%s/SAVE_%i/TEMP_%i ]; then mv -f %s/%s/SAVE_%i/TEMP_%i %s/%s/SAVE_%i/SAVE_%i; fi\n", 
		   storedir, labelstr, (int)(set[i].soln_num/100)*100, 
		   set[i].soln_num, storedir, labelstr, 
		   (int)(set[i].soln_num/100)*100, set[i].soln_num, 
		   storedir, labelstr, (int)(set[i].soln_num/100)*100, 
		   set[i].soln_num);
	  system( cmdstr );
	}
	else
	  fprintf(fopti,"\t)\t%i\n", set[i].rank);
      }
      fprintf(fopti,"\n");
      fflush(fopti);
      
      fprintf(stdout,"Test functions has been evaluated %i times:  parameter set generation %i\n",
	      SOLVE_CNT, try, N_SET);

      // remove all sets that have not been output as part of a generatation
      sprintf( cmdstr, "find %s/%s/ -name 'TEMP_*' -exec rm -r {} \\; ", storedir, labelstr );
      system( cmdstr );
    }

    fprintf(fopti,"==========\nTry %i has Rmax=%i ranks with N_Rmax=%i in the highest rank. Solving...\n", 
              try, Rmax, N_Rmax);
    printf("==========\nTry %i has Rmax=%i ranks with N_Rmax=%i in the highest rank. Solving...\n", 
              try, Rmax, N_Rmax);

    /** Determine simplex sets based on cells with rank == Rmax,
	and randomly selected N_PARAM points from lower ranks **/
    for ( i = 0; i < N_Rmax; i++ ) {
      printf("now trying set %d out of %d (N_Rmax-1)\n",i,N_Rmax-1);
      
      /** Set first parameter set to member of worst rank **/
      for(param=0;param<N_PARAM;param++) {
	test_set[0].p[param] = set[N_SET-i-1].p[param];
      }
      test_set[0].y    = set[N_SET-i-1].y; /* holds Rsqr */
      test_set[0].rank = set[N_SET-i-1].rank;
      test_set[0].pos  = N_SET-i-1;
      for(j=0;j<N_TEST_FUNCS;j++) test_set[0].f[j] = set[N_SET-i-1].f[j];

      iter = 0;

      /* Determine minimum criteria for a non-dominated set */
      mincheck = 0;
      for ( j = 0; j < N_SET; j++ ) {
	if ( set[j].rank == Rmax-1 ) {
	  if ( mincheck == 0 ) {
	    mincheck ++;
	    for ( test = 0; test < N_TEST_FUNCS; test++ ) 
	      Dcheck[test] = set[j].f[test];
	  }
	  else {
	    for ( test = 0; test < N_TEST_FUNCS; test++ ) 
	      if ( set[j].f[test] > Dcheck[test] ) 
		Dcheck[test] = set[j].f[test];
	  }
	}
      }
	
      /* iterate through test sets ------------- */
      printf("now iterating through test sets: iter=%d\n",iter);
      do {
	/** randomly select other simplex points from points/sets with rank
	    less than Rmax.  selection based on probability distribution **/

        /* AWW:  this appears to break down if N_PARAM > (N_SET-N_Rmax)
                 put following check.  If there are fewer sets of rank lower 
                 than the highest (e.g., worst) rank, then do not proceed  */
        if((N_SET-N_Rmax) < N_PARAM) {
  	  printf("EXITING: the current generation of %d sets has fewer sets (only N_SET-N_Rmax=%d) having rank lower than Rmax=%d than the N_PARAM=%d sets required.\nCheck output -- you may have reached an acceptable calibration\n",N_SET, N_SET-N_Rmax, Rmax,N_PARAM);
          exit(5);
	}
        /* AWW:  -------- end additions */
   
	for ( j = 1; j <= N_PARAM; j++ ) {
          cycle=0;
	  do {
	    tmp_prob = ran2(ran2seed);
	    k = 0;
	    while(tmp_prob>prob[k] && k<N_SET-1) k++;
            printf("k=%d\nj=%d cycle=%d ",k,j,cycle);
	    if ( k < (N_SET - N_Rmax) ) {
	      test_set[j].pos = k;
	      NEW_SET=TRUE;
	      for ( l = 0; l < j; l++ ) {
		/*		printf("l=%d j=%d\n",l,j);*/
		if(test_set[l].pos==test_set[j].pos) NEW_SET=FALSE;
              }
	    } else    NEW_SET=FALSE;

            cycle++;
            if(cycle>10000)    exit(0);	/* AWW */

	  } while(!NEW_SET);
	  
	  for ( param = 0; param < N_PARAM; param++ ) 
	    test_set[j].p[param] = set[test_set[j].pos].p[param];
	  test_set[j].y    = set[test_set[j].pos].y;
	  test_set[j].rank = set[test_set[j].pos].rank;
	  for(param=0;param<N_TEST_FUNCS;param++) 
	    test_set[j].f[param] = set[test_set[j].pos].f[param];
	  
	}

	quick(test_set,N_PARAM+1);
	
	for ( j = 0; j <= N_PARAM; j++ ) {
	  for ( param = 0; param < N_PARAM; param++ ) 
	    p[j][param] = test_set[j].p[param]; /* set up square array of
						   parameters ? AWW */
	  y[j] = test_set[j].y;
	  soln_num[j] = test_set[j].soln_num;
	  for ( param = 0; param < N_TEST_FUNCS; param++ ) 
	    f[j][param] = test_set[j].f[param];
	}

	/** solve downhill simplex exactly once **/
	printf("\nRmax(try %i) entering amoeba...\n",i);
	fprintf(fopti,"\tRmax(%i) entering amoeba...\n",i);
	fprintf(fopti,"\t-----> ");
	for ( j = 0; j < N_TEST_FUNCS; j++ ) 
	  fprintf(fopti,"\t%f",test_set[N_PARAM].f[j]);
	fprintf(fopti,"\n");
	
        /* p = parameters, y = Rsqr, f = test functions; FOUND_BETTER = T or F */
	FOUND_BETTER = amoeba( p, y, f, soln_num, N_PARAM, F_TOL, &iter, 
			       runstr, param_lim, labelstr, storedir, 
			       storefiles, fopti);
      
	fprintf(fopti,"\t-----> ");
	for ( j = 0; j < N_TEST_FUNCS; j++ ) 
	  fprintf(fopti,"\t%f",f[N_PARAM][j]);
	fprintf(fopti,"\n");

	/* Check to see if new point is dominated */
	if ( FOUND_BETTER ) {
	  minsum = 0;
	  for ( test = 0; test < N_TEST_FUNCS; test++ ) 
	    if ( f[N_PARAM][test] <= Dcheck[test] ) minsum++;
	  
	  if ( minsum == 0 ) 
	    fprintf(fopti,"WARNING - new point from amoeba still dominated.\n");
	}

      } while ( iter < MAXITER && !FOUND_BETTER );

      /** replace high rank point with simplex solution **/
      for ( param = 0; param < N_PARAM; param++ ) 
	set[N_SET-i-1].p[param] = p[N_PARAM][param];
      set[N_SET-i-1].y = y[N_PARAM]; /* hold Rsqr values */
      set[N_SET-i-1].soln_num = soln_num[N_PARAM];
      for ( j = 0; j < N_TEST_FUNCS; j++ ) 
	set[N_SET-i-1].f[j] = f[N_PARAM][j];

    }

    try++;

    Rmax = rank(set);
    quick(set,N_SET);
    free((char *)prob);
    prob = calc_rank_probs(set,Rmax,&N_Rmax);

  }

  //
  // Optimization has been completed, output final generation
  //

  fprintf(fopti,"\nResults for multi-objective global optimization:\n");
  fprintf(fopti,"\tNeeded %i iterations to solve with a population of %i\n==========\n",try,N_SET);
  /*header row*/
  for ( j = 0; j < N_PARAM; j++) fprintf(fopti,"\t%s",param_lim[j].name);
  for ( j = 0; j < N_TEST_FUNCS; j++ )    fprintf(fopti,"\ttest%d",j+1);
  if ( storefiles ) {
    fprintf(fopti," \trank\tsoln_num\n", set[i].rank, set[i].soln_num);
  }
  fprintf(fopti,"\n");
  /* param & results specs */
  for ( i = 0; i < N_SET; i++ ) {
    fprintf(fopti,"%i:",i);
    for ( j = 0; j < N_PARAM; j++) 
      fprintf(fopti,"\t%.5g",set[i].p[j]);
    fprintf(fopti,"\t= (");
    for ( j = 0; j < N_TEST_FUNCS; j++ )
      fprintf(fopti,"\t%.5g",set[i].f[j]);
    if ( storefiles ) {
      fprintf(fopti," )\t%i\t%i\n", set[i].rank, set[i].soln_num);
      // identify current set for saving
      sprintf( cmdstr, "if [ -d %s/%s/SAVE_%i/TEMP_%i ]; then mv -f %s/%s/SAVE_%i/TEMP_%i %s/%s/SAVE_%i/SAVE_%i; fi\n", 
	       storedir, labelstr, (int)(set[i].soln_num/100)*100, 
	       set[i].soln_num, storedir, labelstr, 
	       (int)(set[i].soln_num/100)*100, set[i].soln_num, 
	       storedir, labelstr, (int)(set[i].soln_num/100)*100, 
	       set[i].soln_num);
      system( cmdstr );
    }
    else
      fprintf(fopti,"\t)\t%i\n", set[i].rank);
  }
  fprintf(fopti,"\n");

  fprintf(stdout,"Optimization required the function be evaluated %i times, through %i generations.\nDONE!\n",
	  SOLVE_CNT,try);
    
  // remove all sets that have not been output as part of a generatation
  sprintf( cmdstr, "find %s/%s/ -name 'TEMP_*' -exec rm -r {} \\; ", storedir, labelstr );
  system( cmdstr );

}


void random_start_optimization(FILE        *fopti,
			       ITEM        *set, 
			       PARAM_RANGE *param_lim,
			       int         *N_Rmax,
			       int          N_RAND,
                               int          N_SET,
			       int          N_TEST_FUNCS,
			       int         *Rmax,
			       float      **prob,
			       long        *ran2seed,
			       char        *runstr,
			       char        *labelstr, 
			       char        *storedir,
			       int          storefiles) {
  /* This routine generates a random population of N_RAND sets, runs each, evaluates
     them, and then culls the set to the best N_SET parameter sets, which is
     uses to kick off the optimization search */

  extern int  SOLVE_CNT;

  int         i, j, param, setcnt;
  char        cmdstr[1024];

  /** formulate the original parameter set **/
  fprintf(fopti,"Determining starting parameters...\n==========\n");
  for(j = 0; j < N_PARAM; j++) fprintf(fopti,"\t%s",param_lim[j].name);
  fprintf(fopti,"\n");

  for ( setcnt = 0; setcnt < N_RAND; setcnt++ ) {
    
    fprintf(fopti,"%i:\t",setcnt);

      for ( param = 0; param < N_PARAM; param++ ) {
	set[setcnt].p[param] = ((param_lim[param].max 
			       - param_lim[param].min)
			      * (ran2(ran2seed))) 
	  + param_lim[param].min;
      }
      
      /* !!!!!! RUN, ROUTE, CALC STATS FOR ONE PARAMETER SET !!!!!!!!!!!!!! */
      set[setcnt].y = solve_model(set[setcnt].p, set[setcnt].f, runstr, 
				  param_lim, labelstr, fopti, storedir, 
				  storefiles);
      set[setcnt].soln_num = SOLVE_CNT;

      if ( storefiles ) {
	// Rename set directory to store it
	sprintf( cmdstr, "if [ -d %s/%s/SAVE_%i/TEMP_%i ]; then mv -f %s/%s/SAVE_%i/TEMP_%i %s/%s/SAVE_%i/SAVE_%i; fi\n", 
		 storedir, labelstr, (int)(set[setcnt].soln_num/100)*100, 
		 set[setcnt].soln_num, storedir, labelstr, 
		 (int)(set[setcnt].soln_num/100)*100, set[setcnt].soln_num, 
		 storedir, labelstr, (int)(set[setcnt].soln_num/100)*100, 
		 set[setcnt].soln_num);
	system( cmdstr );
      }

  }
  
  /** Rank parameter set */
  (*Rmax) = rank( set );   /* Rmax given maximum rank needed to compose test set, where
			      the lower rankings are assigned to better sets */
  
  /* Sort parameter set by ranking */
  quick( set, N_RAND );

  /** Strip out worst random start values **/
  for ( i = 0; i < N_SET; i++ )
    set[i] = set[i+N_RAND-N_SET];  /* store lowest rankings in elements i <
				      N_SET */

  /* Calculate rank probabilities */
  prob[0] = calc_rank_probs( set, *Rmax, N_Rmax );

}


void restart_optimization(char        *filename,
			  FILE        *fopti,
			  ITEM        *set, 
			  PARAM_RANGE *param_lim,
			  int         *N_Rmax,
			  int          N_SET,
			  int          N_TEST_FUNCS,
			  int         *Rmax,
			  float      **prob,
			  int          storefiles) {
  /* This routine reads a population of simulations produced therough
     another optimization attempt or even via a simple random start, it
     then uses these points to restart the optimizer. */

  FILE       *f;
  int         j, param, setcnt, test;
  char        ErrStr[512];

  if ( ( f = fopen(filename,"r") ) == NULL ) {
    sprintf(ErrStr, "Unable to open restart file %s", filename);
    nrerror(ErrStr);
  }

  /** formulate the original parameter set **/
  fprintf(fopti,"Reading starting parameters from %s...\n==========\n", filename);
  for ( j = 0; j < N_PARAM; j++ ) fprintf(fopti, "\t%s", param_lim[j].name);
  fprintf(fopti,"\n");

  for ( setcnt = 0; setcnt < N_SET; setcnt++ ) {
    
    fprintf(fopti, "%i:", setcnt);
    for ( param = 0; param < N_PARAM; param++ ) {
      fscanf(f, "%f", &set[setcnt].p[param]);
      fprintf(fopti, "\t%f", set[setcnt].p[param]);
    }
    fprintf(fopti," \t=\t(");
    for ( test = 0; test < N_TEST_FUNCS; test++ ) {
      fscanf(f, "%f", &set[setcnt].f[test]);
      fprintf(fopti, "\t%f", set[setcnt].p[test]);
    }
    if (storefiles) {
      fscanf(f, "%i", &set[setcnt].soln_num);
      set[setcnt].soln_num *= -1;
      fprintf(fopti, "\t)\t%i\n", set[setcnt].soln_num);
    }
    else
      fprintf(fopti, "\t)\n");
      
  }
  fclose(f);
  
  /** Rank parameter set */
  (*Rmax) = rank( set );
  
  /* Sort parameter set by ranking */
  quick( set, N_SET );

  /* Compute rank probabilities */
  prob[0] = calc_rank_probs( set, *Rmax, N_Rmax );

}

 
int amoeba(float       **p,
	   float        *y,
	   float       **f,
	   int          *soln_num,
	   int           ndim,
	   float         ftol,
	   int          *extern_iter,
	   char         *runstr,
	   PARAM_RANGE  *param_lim,
	   char         *labelstr,
	   char         *storedir,
	   int           storefiles,
	   FILE         *fopti)
{
  /* NOTE:  p = parameters, y = Rsqr, f = test stats, ndim = N_PARAM */
/***********************************************************************
  Simplex optimization routine from Numerical Recipies

  Modifications:
  11-18-99 Modifed so that the routine returns the first value, 
           instead of iterating to find the "best" value (as 
	   described by Yapo et al. 1998).
  01-05-01 Modified to record the solution number as reference
           for stored information.

***********************************************************************/
  extern int SOLVE_CNT;

  int    mpts,j,inhi,ilo,ihi,i,k;
  int    pr_soln_num, prr_soln_num, prrr_soln_num;
  int    FOUND;
  float  yprrr, yprr, ypr;  /* these hold Rsqr returned by solve_model() */
  float *fprrr, *fprr, *fpr;
  float *pr, *prr, *prrr, *pbar;
  float  *storef;
  
  pr     = (float *)calloc(ndim,sizeof(float));  /* these are all param arrays */
  prr    = (float *)calloc(ndim,sizeof(float));
  prrr   = (float *)calloc(ndim,sizeof(float));
  pbar   = (float *)calloc(ndim,sizeof(float));
  fpr    = (float *)calloc(N_TEST_FUNCS,sizeof(float)); /* these are all stats
							   arrays */
  fprr   = (float *)calloc(N_TEST_FUNCS,sizeof(float));
  fprrr  = (float *)calloc(N_TEST_FUNCS,sizeof(float));
  storef = (float *)calloc(N_TEST_FUNCS,sizeof(float));

  FOUND  = FALSE;
  mpts   = ndim+1;
  ihi    = ndim;   /* = N_PARAM */
  inhi   = ndim-1;
  ilo    = 0;
  for ( k = 0; k < N_TEST_FUNCS; k++ ) storef[k] = f[ihi][k];
  extern_iter[0]++;
  
  /* AWW add */ printf("Starting amoeba w/ initial stats:  ");
  /* AWW add */ for ( k = 0; k <  N_TEST_FUNCS; k++ ) printf("\t%g", storef[k]);
  /* AWW add */ printf("\n");
  
  /** Reflect simplex from the high point ---------------------- **/
  
  /* generate first modified parameter set for testing?  AWW */
  for ( j = 0; j < ndim; j++ ) pbar[j] = 0.0;
  for ( i = 0; i < mpts; i++ )
    if (i != ihi)
      for ( j = 0; j < ndim; j++ ) pbar[j] += p[i][j];
  for ( j = 0; j < ndim; j++ ) {
    pbar[j] /= ndim;
    pr[j] = (1.0+ALPHA)*pbar[j] - ALPHA*p[ihi][j];
    /* HERE'S WHERE TO ADD THE TOLERANCES (AND IN LIKE PARAM MODIFS BELOW) 
       
       NEED ROUNDING FUNCTION - IT SHOULD CHECK DIRECTION OF MODIFS, AND TAKE PARAMS
       TO THE NEXT TOLERANCE UNIT IN THAT DIRECTION - AWW */
  }

  /* AWW add */ printf("and test ALPHA modified params:  ");
  /* AWW add */ for ( j = 0; j < ndim; j++ ) printf("\t%g", pr[j]);
  /* AWW add */ printf("\n");

  /* !!!!!! RUN, ROUTE, CALC STATS FOR ONE PARAMETER SET !!!!!!!!!!!!!! */
  /* pr = parameters, fpr = test stats array */
  ypr = solve_model( pr, fpr, runstr, param_lim, labelstr, fopti, 
		     storedir, storefiles );
  pr_soln_num = SOLVE_CNT;

  if ( less_than_or_equal(fpr,f[ilo],N_TEST_FUNCS) ) {
    
    /** Solution better than best point, so try additional 
	extrapolation by a factor of GAMMA **/
    
    /* generate 2nd modified parameter set in same 'direction'?  AWW */
    for ( j = 0; j < ndim; j++ )
      prr[j] = GAMMA*pr[j] + (1.0-GAMMA)*pbar[j];

    /* !!!!!! RUN, ROUTE, CALC STATS FOR ONE PARAMETER SET !!!!!!!!!!!!!! */
    yprr = solve_model( prr, fprr, runstr, param_lim, labelstr, fopti, 
			storedir, storefiles );
    prr_soln_num = SOLVE_CNT;
    
    if ( less_than(fprr,f[ilo],N_TEST_FUNCS) ) {
      /* Use additional extrapolation value since better stats than previous*/
      for ( j = 0; j < ndim; j++ ) p[ihi][j] = prr[j];
      y[ihi] = yprr;
      soln_num[ihi] = prr_soln_num;
      for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fprr[j];
      FOUND = TRUE;
    } else {
      /* Additional extrpolation not as good, use original reflection */
      for ( j = 0; j < ndim; j++ ) p[ihi][j] = pr[j];
      y[ihi]=ypr;
      soln_num[ihi] = pr_soln_num;
      for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fpr[j];
      FOUND = TRUE;
    }
    
  } else if ( less_than_or_equal(f[inhi],fpr,N_TEST_FUNCS) ) {
    
    /* if reflected point is larger than the second largest point,
       look for an intermediate lower point by doing a one-dimensional 
       contraction */
    
    if ( less_than(fpr,f[ihi],N_TEST_FUNCS) ) {
      /* Replace original high point with reflection, if reflection
	 is better */
      for ( j = 0; j < ndim; j++ ) p[ihi][j] = pr[j];
      y[ihi] = ypr;
      soln_num[ihi] = pr_soln_num;
      for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fpr[j];
      FOUND = TRUE;
    }
    
    /* contract */
    /* generate alternate 2nd modified parameter set?  AWW */
    for ( j = 0; j < ndim; j++ )
      prr[j] = BETA*p[ihi][j] + (1.0-BETA)*pbar[j];

    /* !!!!!! RUN, ROUTE, CALC STATS FOR ONE PARAMETER SET !!!!!!!!!!!!!! */
    yprr = solve_model( prr, fprr, runstr, param_lim, labelstr, fopti, 
			storedir, storefiles );
    prr_soln_num = SOLVE_CNT;
    
    if ( less_than(fprr,f[ihi],N_TEST_FUNCS) ) {
      
      /* Contraction yielded smaller point, store it */
      
      for ( j = 0; j < ndim; j++ ) p[ihi][j] = prr[j];
      y[ihi] = yprr;
      soln_num[ihi] = prr_soln_num;
      for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fprr[j];
      FOUND = TRUE;

    } else {
      
      /* Contraction did not yield smaller point, 
	 try contracting from all sides */
      
      /* generate 3rd modified parameter set?  AWW */
      for ( i = 0; i < mpts; i++ ) {
	if ( i != ilo ) {
	  for ( j = 0; j < ndim; j++ ) {
	    pr[j] = 0.5*(p[i][j] + p[ilo][j]);
	    prrr[j] = pr[j];
	  }

          /* !!!!!! RUN, ROUTE, CALC STATS FOR ONE PARAMETER SET !!!!!!!!!!!!!! */
	  yprrr = solve_model( prrr, fprrr, runstr, param_lim, labelstr, fopti, 
			      storedir, storefiles );
	  prrr_soln_num = SOLVE_CNT;
	  for ( k = 0; k < N_TEST_FUNCS; k++ ) f[i][k] = fpr[k];
	}
	if ( less_than(fprrr,f[ihi],N_TEST_FUNCS) ) {
	  
	  /* Contraction from all sides yielded smaller point, store it */
	  
	  for ( j = 0; j < ndim; j++ ) p[ihi][j] = prrr[j];
	  y[ihi] = yprrr;
	  soln_num[ihi] = prrr_soln_num;
	  for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fprrr[j];
	  FOUND = TRUE;
	} 
      }
    }

  } else {
    
    /** Reflection yielded a lower high point **/ 
    
    for ( j = 0; j < ndim; j++ ) p[ihi][j] = pr[j];
    y[ihi] = ypr;
    soln_num[ihi] = pr_soln_num;
    for ( j = 0; j < N_TEST_FUNCS; j++ ) f[ihi][j] = fpr[j];
    FOUND = TRUE;

  }

  free((char*)pbar);
  free((char*)prrr);
  free((char*)prr);
  free((char*)pr);
  free((char*)fprrr);
  free((char*)fprr);
  free((char*)fpr);

  return(FOUND);

}

float solve_model(float       *p, 
		  float       *f,
		  char        *runstr, 
		  PARAM_RANGE *param_lim, 
		  char        *labelstr,
		  FILE        *fopti,
		  char        *storedir,
		  int          storefiles) {
/* NOTE: p = parameters; f = test stats; function returns Rsqr */
/***********************************************************************
  solve_model

  This subroutine checks the parameters in *p to see if they are valid.
  If so, then it starts the script that runs the model with the new
  parameter set.  Before returning to the amoeba it opens and reads 
  the R squared value for the run just completed, so that it can be 
  returned.  Invalid parameters return a default value of INVALIDMAX.

  Modifications:
  010501 Modified to record all 

***********************************************************************/

  extern int SOLVE_CNT;

  FILE *fin;
  char *cmdstr;
  char  BADPARAM;
  char  filename[512];
  char  tmpstr[1024];
  char  ErrStr[512];
  float Rsqr;
  int   i;

  BADPARAM = FALSE;
  for(i=0;i<N_PARAM;i++) 
    if((p[i]<param_lim[i].min) || (p[i]>param_lim[i].max)) 
      BADPARAM = TRUE;

  if(!BADPARAM) {

    SOLVE_CNT++;

    cmdstr = (char *)calloc(512,sizeof(char));
    
    sprintf(filename,"R2_and_teststats.%s.txt",labelstr);
    strcpy(cmdstr,runstr);
    for(i=0;i<N_PARAM;i++) {
      sprintf(tmpstr," %f",p[i]);
      strcat(cmdstr,tmpstr);
    }
    strcat(cmdstr," ");
    strcat(cmdstr,filename);
    if ( storefiles ) {
      sprintf(tmpstr," %s/%s/SAVE_%i/TEMP_%i", storedir, labelstr, 
	      SOLVE_CNT, SOLVE_CNT);
      /*	      int(SOLVE_CNT/100)*100, SOLVE_CNT); */
      strcat(cmdstr,tmpstr);
    }
    
    system(cmdstr);

    if((fin=fopen(filename,"r"))==NULL) {
      sprintf(ErrStr,"Unable to open Rsquared file %s", filename);
      nrerror(ErrStr);
    }
    
    fscanf(fin,"%f",&Rsqr);
    for ( i = 0; i < N_TEST_FUNCS; i++ ) fscanf(fin,"%f",&f[i]);
    fclose(fin);
    for ( i = 0; i < N_PARAM; i++ )
      fprintf(fopti,"%f\t",p[i]);
    fprintf(fopti,"=\t(");
    for ( i = 0; i < N_TEST_FUNCS; i++ ) fprintf(fopti,"\t%f",f[i]);
    if ( storefiles )
      fprintf(fopti,"\t)\t-1\t%i\n", SOLVE_CNT);
    else
      fprintf(fopti,"\t)\t-1\n");
    
    free((char *)cmdstr);
    
  }
  else {
    
    Rsqr = (float)INVALIDMAX;
    for(i=0;i<N_TEST_FUNCS;i++) f[i] = (float)INVALIDMAX;

    fprintf(fopti,"Invalid Parameter Set:\n");
    for(i=0;i<N_PARAM;i++) 
      fprintf(fopti,"%f\t",p[i]); 
    fprintf(fopti,"-> returning %f\n",Rsqr);
    
  }
  fflush(fopti);

  return(Rsqr);

}


void nrerror(char *errstr) {
/********************************************************************
  Error handling routine
********************************************************************/
  fprintf(stderr,"NRERROR: %s\n",errstr);
  exit(0);
}


PARAM_RANGE *set_param_limits(char *fname, int Nparam) {
/**************************************************************
  Sets limits for all parameters
**************************************************************/

  FILE *fin;
  PARAM_RANGE *param_lim;
  int     i;

  param_lim = (PARAM_RANGE *)calloc(Nparam, sizeof(PARAM_RANGE));

  if((fin=fopen(fname,"r"))==NULL) 
    nrerror("Unable to open parameter range file.");

  for( i = 0; i < Nparam; i++ ) {
    fscanf(fin,"%s %f %f", param_lim[i].name, &param_lim[i].max, 
	   &param_lim[i].min);
  }

  return(param_lim);

}

int rank(ITEM *set) {
/*********************************************************************
  This routine loops through the population of solution sets, and 
  ranks each group of non-dominated solutions.  Ranking starts by
  finding the globally non-dominated sets (rank 1).  Rank 1 sets
  are removed, and the next set of non-dominated solutions is found.
  This is repeated until all solution sets have been ranked.

  see Goldberg, "Genetic Algorithms in Search, Optimization, and
  Machine LEarning," 1989, for a discussion of non-dominated sets.
*********************************************************************/

  extern int N_SET;

  int i, cnt, rank, N, lastN, test, MIN;

  lastN = 0;
  N = 0;
  rank = 1;
  for(cnt=0;cnt<N_SET;cnt++) {
    set[cnt].rank = 0;  /* initialize rank for all sets to zero */
  }
        
  /** Repeat Ranking Until All Population Members have a rank **/
  while(N < N_SET) {

    for(cnt=0;cnt<N_SET;cnt++) {  /* cycle solution sets */
      if(set[cnt].rank==0) {
	/** if set unranked, check if set dominated **/
	if(N<N_SET-1) { /* for all but last set... */
	  for(i=0;i<N_SET;i++) { /* compare set[cnt] against set[i] */
	    if((set[i].rank==0 || set[i].rank==rank) && i!=cnt) {
              /* if set[i] has rank 0 or current rank, test it  */
	      MIN = 0;
	      for(test=0;test<N_TEST_FUNCS;test++) {
                /* if set[cnt] <= set[i] on at least one test, it's not
		   dominated at current rank */
		if(set[cnt].f[test] <= set[i].f[test]) MIN++;
	      }
	      if(MIN==0) break; /** Current point dominated by test point **/
	    }
	  }
	  if(MIN!=0) {
	    set[cnt].rank = rank;  /* last set gets current rank */
	    N++;
	  }
	}
	else {
	  /** Only one unrankeds member left **/
	  set[cnt].rank = rank;
	  N++;
	}
      }
    }
    /* now have done all set by set comparisons and set those not dominated at
       current rank (1 being best); move on to next ranks */
    rank++;
    /*    if(rank>N_SET) {  this could be a bug: if all sets were ranked and
	  each set was assigned a different rank, this can logically occur.
	  Modified so exclude that possiblity:  AWW */
    if(rank>N_SET && N!=N_SET) {
      printf("\n%d sets ranked out of %d\n",N,N_SET);
      /* possibly not always an error */
      fprintf(stderr,"ERROR ======= rank %d is greater than N_SET %d\n", rank, N_SET);
      fprintf(stderr,"set#                  teststats                  ->  rank\n");
      for(cnt=0;cnt<N_SET;cnt++) {
	fprintf(stderr,"%i:\t",cnt);
	for(test=0;test<N_TEST_FUNCS;test++) 
	  fprintf(stderr,"%f\t",set[cnt].f[test]);
	fprintf(stderr,"->\t%i\n",set[cnt].rank);
	if(set[cnt].rank==0) set[cnt].rank=rank;
      }
      rank++;
    }
  }

  if(N!=N_SET) {
    fprintf(stderr,"ERROR: %i population members were not ranked.\n",N_SET-N);
    exit(0);
  }

  printf("returning rank %d at end of rank()\n",rank-1);
  return(rank-1);

}

float *calc_rank_probs(ITEM *set, int Rmax, int *N_Rmax) {
/**********************************************************************
  this routine computes the probability that each rank will produce
  offspring.  the lowest ranks have the highest probability of
  reproducing.
**********************************************************************/

  int    i, j;
  float *prob;
  float  sum;

  /** Compute rank probabilities **/

  prob = (float *)malloc(N_SET*sizeof(float));
  *N_Rmax = 0;

  for ( i = 0; i < N_SET; i++ ) {
    sum = 0;
    for ( j = i; j < N_SET; j++ ) sum += set[j].rank;
    set[i].prob = (Rmax - set[i].rank + 1) / (N_SET * (Rmax + 1) - sum);
    if ( i == 0 ) prob[i] = set[i].prob;
    else prob[i] = set[i].prob + prob[i-1];
    if ( set[i].rank == Rmax ) N_Rmax[0]++;
  }

  return(prob);

}

void quick(ITEM *item, int count)
/**********************************************************************
        this subroutine starts the quick sort
**********************************************************************/
{
  qs(item,0,count-1);
}
 
void qs(ITEM *item, int left, int right)
/**********************************************************************
        this is the quick sort subroutine - it returns the values in
	an array from high to low.
**********************************************************************/
{
  register int i,j;
  ITEM x,y;

  i=left;
  j=right;
  x=item[(left+right)/2];
 
  do {
    while(item[i].rank<x.rank && i<right) i++;
    while(x.rank<item[j].rank && j>left) j--;
 
    if (i<=j) {
      y=item[i];
      item[i]=item[j];
      item[j]=y;
      i++;
      j--;
    }
  } while (i<=j);
 
  if(left<j) qs(item,left,j);
  if(i<right) qs(item,i,right);

}

#define M 714025
#define IA 1366
#define IC 150889
 
float ran2(long *idum)
/******************************************************************
  Random number generator from Numerical Recipes
******************************************************************/
{
        static long iy,ir[98];
        static int iff=0;
        int j;
 
        if (*idum < 0 || iff == 0) {
                iff=1;
                if ((*idum=(IC-(*idum)) % M) < 0) *idum = -*idum;
                for (j=1;j<=97;j++) {
                        *idum=(IA*(*idum)+IC) % M;
                        ir[j]=(*idum);
                }
                *idum=(IA*(*idum)+IC) % M;
                iy=(*idum);
        }
        j=1 + 97.0*iy/M;
        if (j > 97 || j < 1) nrerror("RAN2: This cannot happen.");
        iy=ir[j];
        *idum=(IA*(*idum)+IC) % M;
        ir[j]=(*idum);
        return (float) iy/M;
}
 
#undef M
#undef IA
#undef IC
#undef ALPHA
#undef BETA
#undef GAMMA
#undef ITMAX
#undef INVALIDMAX

void code_error(char *errstr) {

  fprintf(stderr,"ERROR: ");
  fprintf(stderr,errstr);
  fprintf(stderr,"\n");
  exit(0);

}

int less_than(float *x, float *y, int n) {
  /* counts items in first array that are less than corresponding items in 2nd
     array */
  int i, cnt;
  cnt=0;
  for(i=0;i<n;i++) if(x[i]<y[i]) cnt++;
  return cnt;
}

int less_than_or_equal(float *x, float *y, int n) {
  /* counts items in first array that are less than OR EQUAL TO corresponding
     items in 2nd array */
  int i, cnt;
  cnt=0;
  for(i=0;i<n;i++) if(x[i]<=y[i]) cnt++;
  return cnt;
}

void usage() {
  fprintf(stderr,"\nUSAGE: need arguments: <num start | start file> <num sets> <num param> <num tests> <model run script> <model run identifier> <optimization log> <parameter range file> [<storage directory>]\n");
    fprintf(stderr,"\n\tThis program uses the MOCOM-UA multi-objective optimizing scheme to\n\toptimize anything included in the optimization script that returns\n\tstatistics that can be minimized to find the 'actual' solution.\n\tThe result will be a set of simulations that define the Pareto solution\n\tset.  Given a sufficiently large population, the Pareto set should\n\tdefine the best set of solutions obtainable with the calibration\n\tscript.\n");
    fprintf(stderr,"\n\t<num start | start file> number of random simulation parameter sets\n\t\twith which to start the test population or the name of restart\n\t\tfile.\n");
    fprintf(stderr,"\t<num sets> number of simulation parameter sets in the test population.\n");
    fprintf(stderr,"\t<num param> number of parameters used by the simulation.\n");
    fprintf(stderr,"\t<num tests> number of test functions to be minimized.\n");
    fprintf(stderr,"\t<model run script> is the model run script to be used to control the\n\t\tmodel simulations.\n");
    fprintf(stderr,"\t<model run identifier> is a character string used to separate run\n\t\ttime files created by this program, from those created by other\n\t\tsimultaneous runs of this program.\n");
    fprintf(stderr,"\t<optimization log> is a log file which records the steps the\n\t\toptimizer has taken, and the final optimized results.\n");
    fprintf(stderr,"\t<parameter range file> is a test file, each line of which gives the\n\t\tparameter name, maximum and minimum values.\n");
    fprintf(stderr,"\t<storage directory> is the directory in which discharge and parameter\n\t\tinformation for all simulations will be stored.  If no directory\n\t\tis provided than no discharge information will be stored.\n\t\tSimulations are stored and number sequentially, simulation number\n\t\tis stored in the log file.\n\n");
    exit(0);
}
