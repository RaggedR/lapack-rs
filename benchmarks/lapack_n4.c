#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
static double now(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(){
  int n=4;
  double *A=malloc(n*n*8),*B=malloc(n*n*8),*C=malloc(n*n*8),*x=malloc(n*8),*y=malloc(n*8),*S=malloc(n*n*8);
  for(int i=0;i<n*n;i++){A[i]=rand()/(double)RAND_MAX;B[i]=rand()/(double)RAND_MAX;}
  for(int i=0;i<n;i++){x[i]=rand()/(double)RAND_MAX;y[i]=rand()/(double)RAND_MAX;}
  for(int i=0;i<n;i++)for(int j=0;j<n;j++)S[i*n+j]=A[i*n+j]+A[j*n+i]; // symmetric
  int R=2000000;
  double t;
  t=now(); for(int r=0;r<R;r++){volatile double v=cblas_ddot(n,x,1,y,1);(void)v;} double td=(now()-t)/R;
  t=now(); for(int r=0;r<R;r++) cblas_dgemm(CblasColMajor,CblasNoTrans,CblasNoTrans,n,n,n,1.0,A,n,B,n,0.0,C,n); double tm=(now()-t)/R;
  int lwork=n*64,info; double *tau=malloc(n*8),*work=malloc(lwork*8),*Aq=malloc(n*n*8);
  t=now(); for(int r=0;r<R;r++){for(int i=0;i<n*n;i++)Aq[i]=A[i]; dgeqrf_(&n,&n,Aq,&n,tau,work,&lwork,&info);} double tq=(now()-t)/R;
  double *w=malloc(n*8); char jobz='V',uplo='U';
  t=now(); for(int r=0;r<R;r++){for(int i=0;i<n*n;i++)Aq[i]=S[i]; dsyev_(&jobz,&uplo,&n,Aq,&n,w,work,&lwork,&info);} double te=(now()-t)/R;
  printf("LAPACK n=4:  dot=%.4f us  matmul=%.4f us  qr=%.4f us  eig=%.4f us\n", td*1e6,tm*1e6,tq*1e6,te*1e6);
  return 0;
}
