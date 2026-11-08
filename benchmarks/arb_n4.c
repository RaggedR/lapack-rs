#include <flint/arb.h>
#include <flint/arb_mat.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
int main(){
  slong prec=64; int n=4, R=500000;
  arb_ptr x=_arb_vec_init(n), y=_arb_vec_init(n);
  for(int i=0;i<n;i++){arb_set_d(x+i,rand()/(double)RAND_MAX);arb_set_d(y+i,rand()/(double)RAND_MAX);}
  arb_t res; arb_init(res);
  double t=now(); for(int r=0;r<R;r++) arb_dot(res,NULL,0,x,1,y,1,n,prec); double td=(now()-t)/R;
  arb_mat_t A,B,C; arb_mat_init(A,n,n);arb_mat_init(B,n,n);arb_mat_init(C,n,n);
  for(int i=0;i<n;i++)for(int j=0;j<n;j++){arb_set_d(arb_mat_entry(A,i,j),rand()/(double)RAND_MAX);arb_set_d(arb_mat_entry(B,i,j),rand()/(double)RAND_MAX);}
  t=now(); for(int r=0;r<R;r++) arb_mat_mul(C,A,B,prec); double tm=(now()-t)/R;
  arb_mat_t LU; arb_mat_init(LU,n,n); slong*perm=malloc(n*sizeof(slong));
  t=now(); for(int r=0;r<R;r++){arb_mat_lu(perm,LU,A,prec);} double tl=(now()-t)/R;
  printf("Arb (prec=64) n=4:  dot=%.4f us  matmul=%.4f us  LU=%.4f us\n",td*1e6,tm*1e6,tl*1e6);
  return 0;
}
