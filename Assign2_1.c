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

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		printf("Only one argument allowed");
		return -1;
	}
	
	int N, p, myid, ndim, rank, root, sqrt_p;
	int dims[2], coords[2],  mycoords[2], cyclic[2], reorder;
	int i, j, blksqr, blk_size;
	double *A, *B, *C, dx, dy, width;
	int row[2], col[2];
	MPI_Comm comm2D, cart_row, cart_col;
	MPI_Datatype blk;
	MPI_Request *requestA, *requestB, requestC,request[2];
	MPI_Status status[2], statusC;
	p = atoi(argv[1]);
	N = atoi(argv[2]);
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &p);
	
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
	blk_size = N/sqrt_p;
	blksqr = blk_size*blk_size;
	dims[0] = sqrt_p;
	dims[1] = sqrt_p;
	
	/* Creating processor grids.*/
	MPI_Type_vector(blk_size,blk_size, N, MPI_DOUBLE, &blk);
	MPI_Type_commit(&blk);
	MPI_Cart_create(MPI_COMM_WORLD, ndim, dims, cyclic, reorder, &comm2D);
	MPI_Comm_rank(comm2D, &myid);
	MPI_Cart_coords(comm2D, myid, ndim, mycoords);
	
	
	if (myid == 0)
	{
		double centerx, centery, width, cs;
		centerx = round(N/2.0);
		centery = round(N/2.0);
		double *U;
		U = (double*)malloc(N*N*sizeof(double*));
		width = 0.1;
		for (i = 0; i < N; i++)
		{
			for (j = 0; j < N; j++)
			{
				dist = sqrt((j*dx - centerx)*(j*dx - centerx);
					/*(i*dy - centery)*(i*dy - centery));*/
				if (dist < width)
				{
					U[i*N + j]  = cos(M_PI_2*dist/width)*cos(M_PI_2*dist/width);
				}
				else 
				{
					U[i*N + j]  = 0;
				}
				

			}
			U[i] = 0;
			U[i*N] = 0;
			U[i*N + N-1] = 0;
			U[N*(N-1) + i] = 0;
		}
	}
}