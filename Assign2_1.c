/*
 * Partial and distributed programming 
 * Assignment 2. May 10 2017
 * Authors:
 * Jonas Melander
 * Lina Viklund
 * Aleksandra Obeso Duque
 * 
*/
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
	if (argc != 4)
	{
		printf("Only one argument allowed");
		return -1;
	}
	
	int Nx, Ny, p, myid, ndim, rank, root, sqrt_p, T;
	int dims[2], coords[2],  mycoords[2], cyclic[2], reorder;
	int i, j, k, blksqr, blksize;
	double *A, *B, *C, dx, dy, width, lambda_sqr, dt;
	int row[2], col[2];
	MPI_Comm comm2D, cart_row, cart_col;
	MPI_Datatype blk;
	MPI_Request *requestA, *requestB, requestC,request[2];
	MPI_Status status[2], statusC;
	p = atoi(argv[1]);
	Nx = atoi(argv[2]);
	Ny = atoi(argv[3]);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
	
	T = 120;
	dx = 1.0 / (Nx-1);
	dt = 0.5*dx;
	lambda_sqr = (dt/dx)*(dt/dx);
	row[0] = 0;
	row[1] = 1;
	col[0] = 1;
	col[1] = 0;
	ndim = 2;
	root = 0;
	cyclic[0] = 0;
	cyclic[1] = 0;
	reorder = 1;
	sqrt_p = sqrt(p);
	blksize = Nx/sqrt_p;
	blksqr = blksize*blksize;
	dims[0] = sqrt_p;
	dims[1] = sqrt_p;
	
	/* Creating processor grids.*/
	MPI_Type_vector(blksize,blksize, Nx, MPI_DOUBLE, &blk);
	MPI_Type_commit(&blk);
	MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, cyclic, reorder, &comm2D);
	MPI_Comm_rank(comm2D, &myid);
	MPI_Cart_coords(comm2D, myid, ndim, mycoords);
	
	
	//if (myid == 0)
	//{
		double centerx, centery, cs;
		centerx = round(Nx/2.0);
		centery = round(Ny/2.0);
		double *u, x, y, dist;
		u = (double*)malloc(Nx*Ny*sizeof(double*));
		width = 0.1;
		for (i = 0; i < Ny; i++)
		{
			for (j = 0; j < Nx; j++)
			{
				x = j*dx;
				y = j*dx;
				dist = sqrt((x - centerx)*(y - centerx));
					/*(i*dy - centery)*(i*dy - centery));*/
				if (dist < width)
				{
					u[i*Ny + j]  = cos(M_PI_2*dist/width)*cos(M_PI_2*dist/width);
				}
				else 
				{
					u[i*Ny + j]  = 0;
				}
				

			}
			u[i] = 0;
			u[i*Ny] = 0;
			u[i*Ny + Ny-1] = 0;
			u[Ny*(Ny-1) + i] = 0;
		}
	//}
	double *u_new, *u_old;
	u_old= (double*)malloc(blksqr*sizeof(double));
	u_new = (double*)malloc(blksqr*sizeof(double));
	int t;
	for (t = 2; t < T; k++)
	{
		double *tmp = u_old;
		u_old = u;
		u = u_new;
		u_new = tmp;
		for (i = 1; i < blksize; i++)
		{
			for (j = 1; j < blksize; j++)
			{
				u_new[i*blksize + j] = 2*u[i*blksize + j] - u_old[i*blksize + j] + \
				lambda_sqr*(u[(i-1)*blksize + j] + u[(i+1)*blksize + j] + \
							u[i*blksize + j-1] + u[i*blksize + j+1] - \
							4*u[i*blksize + j]);
			} 
		}
		save_solution(u_new, Ny, Nx, t);
	}
	
	free(u);
	free(u_old);
	free(u_new);

	MPI_Finalize();
	
}
