/* MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
             Test Program for Matrix Multiplicaton With Pipelining

   Copyright (c) 2013, 2014, 2017, 2018 Radford M. Neal.

   The matprod library is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/

#include <stdlib.h>
#include <stdio.h>
#include "helpers-app.h"


char *prog_name = "piped-matprod-test";

char * my_var_name (helpers_var_ptr v)
{ char *name = malloc(13);
  if (v>0) sprintf(name,"O%d",v);
  else sprintf(name,"P%d",-v);
  return name;
}  

static int repeat;
static int split;

void do_test (int rep)
{
  char *h = getenv("HELPERS");
  char *t = getenv("TRACE");
  char *s = getenv("SPLIT");
  repeat = rep;
  split = s==0 ? 1 : atoi(s);
  helpers_trace (t!=0);
  helpers_startup (h==0 ? 0 : atoi(h));
}

volatile int vola;

#define chunk 17

void task_output_vector (helpers_op_t op, helpers_var_ptr out,
                         helpers_var_ptr x, helpers_var_ptr y)
{
  double *o = REAL(out);
  int l = LENGTH(out);
  int i, j, k;
  for (k = 0; k<l; k++) 
  { o[k] = -o[k];
  }
  for (k = 0; k+chunk<l; k += chunk)
  { for (i = 0; i<k%9; i++) vola += 1;
    for (j = 0; j<chunk; j++) o[k+j] = -o[k+j];
    helpers_amount_out(k+chunk);
  }
  for (; k<l; k++) 
  { o[k] = -o[k];
  }
}

void helpers_master (void)
{
  int r, i, v;

  for (r = 0; r<repeat; r++)
  { 
    if (vec[nmat])
    { helpers_do_task (HELPERS_PIPE_OUT, task_output_vector, 0, nmat, 0, 0);
    }

    v = vec[nmat];
    for (i = nmat-2; i>=0; i--)
    { int n = matrows[i];
      int k = matcols[i];
      int w;
      if (trans[i] && trans[i+1])
      { fprintf(stderr,"Transposing both operands is not allowed\n");
        exit(1);
      }
      v |= vec[i+1];
      if (vec[i] && v && matrows[i]==1 && matcols[nmat-1]==1) 
      { par_matprod_vec_vec (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), split);
      }
      else if (vec[i] && matrows[i]==1)
      { par_matprod_vec_mat (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), split);
      }
      else if (v && matcols[nmat-1]==1)
      { par_matprod_mat_vec (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), split);
      }
      else if (trans[i])
      { par_matprod_trans1 (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), k, split);
      }
      else if (i==nmat-2 && trans[i+1])
      { par_matprod_trans2 (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), k, split);
      }
      else if (matcols[i]==1 && matrows[nmat-1]==1)
      { par_matprod_outer (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), split);
      }
      else
      { par_matprod_mat_mat (-(i+1), i+1, i+2==nmat ? nmat : -(i+2), k, split);
      }
    }
    helpers_wait_for_all();
  }  

  print_result();

  if (do_check)
  { check_results();
  }
}
