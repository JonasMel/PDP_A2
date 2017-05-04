#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <time.h>

void save_solution(double *u, int Ny, int Nx, int n)
{
  char fname[50];
  sprintf(fname,"solution-%d.dat",n);
  FILE *fp = fopen(fname,"w");

  fprintf(fp,"%d %d\n",Nx,Ny);

  for(int j = 0; j < Ny; ++j) {
    for(int k = 0; k < Nx; ++k) {
      fprintf(fp,"%e\n",u[j*Nx+k]);
    }
  }
}


int main(int argc, char **argv)
{
	T = 120;
	dx = 1.0 / (Nx-1);
	dt = 0.5*dx;
	lambda_sqr = (dt/dx)*(dt/dx);
}