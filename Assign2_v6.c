// mpicc -g -O3 -o wave Assign2_v4.c -lmpi -lm
// mpirun -np 16 ./wave 8 4 4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

#define VERIFY

double timer();
double initialize(double x, double y, double t);

int main(int argc, char **argv)
{
	if (argc != 4)
		return -1;
		
	int Nx, Ny, Nt, px, py, i, j, k, left, right, top, bot, remain_x = 0, remain_y = 0;
	double dt, dx, lambda2, *u, *u_old, *u_new;
	int blk_x, blk_y, blk_y_const, blk_x_const, offset_x, offset_y;
	
	Nx = atoi(argv[1]);
	px = atoi(argv[2]);
	py = atoi(argv[3]);
	Ny = Nx;
	Nt = Nx;
	dx = 1.0 / (Nx-1);
	dt = 0.5*dx;
	lambda2 = (dt/dx)*(dt/dx);
	
	/*Initializing MPI and creating cartesian grid for processors*/
	MPI_Request *send_req, *recv_req, rcv_req[2];
	MPI_Status status[2];
	send_req = (MPI_Request*)malloc(4*sizeof(MPI_Request));
	recv_req = (MPI_Request*)malloc(4*sizeof(MPI_Request));
	MPI_Init(&argc, &argv);
	int ndims = 2;
	int glob_rank, reorder, crt_rank, mycrds[ndims]; 
	int dims[] = {py, px};
	int cyclic[] = {0, 0};
	reorder = 0;
	
	MPI_Comm_rank(MPI_COMM_WORLD, &glob_rank);
	MPI_Comm com_crt, crt_col, crt_row;
	MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &com_crt);
	MPI_Comm_rank(com_crt, &crt_rank);
	MPI_Cart_coords(com_crt, crt_rank, ndims, mycrds);
	
	/*Splitting cartesian grid into columns and rows*/
	MPI_Comm_split(com_crt, mycrds[0], mycrds[1], &crt_row);
	MPI_Comm_split(com_crt, mycrds[1], mycrds[0], &crt_col);
	
	/*Creating vector types for communication*/
	MPI_Datatype row, col;
	
	blk_x = Nx/px;
	remain_x = Nx%px;
	blk_x_const = blk_x;
	if (remain_x > 0 && mycrds[1] < remain_x)
		blk_x++;
		
	blk_y = Ny/py;
	remain_y = Ny%py;
	blk_y_const = blk_y;
	if (remain_y > 0 && mycrds[0] < remain_y)
		blk_y++;
		
	//printf("P: %d, blk_x: %d, blk_y: %d\n", glob_rank, blk_x, blk_y);

	/*Checking location*/
	left = 1;
	right = 1;
	top = 1;
	bot = 1;
	if (mycrds[0] == 0)
		top = 0;
	
	if (mycrds[1] == px-1)
		right = 0;
	
	if (mycrds[1] == 0)
		left = 0;
		
	if (mycrds[0] == py-1)
		bot = 0;

	int top_bnd = 0;
	int bot_bnd;
	if (remain_y > 0 && mycrds[0] < remain_y)
		bot_bnd = blk_y_const;
	else
		bot_bnd = blk_y_const-1;
	int left_bnd = 0;
	int right_bnd;
	if (remain_x > 0 && mycrds[1] < remain_x)
		right_bnd = blk_x_const;
	else
		right_bnd = blk_x_const-1;
	
	if (!top && !left && right && bot)
	{
		blk_x++;
		blk_y++;
		offset_x = 0;
		offset_y = 0;
		top_bnd++;
		left_bnd++;
	}
	else if (!top && left && right & bot)
	{
		blk_x +=2;
		blk_y++;
		offset_x = 1;
		offset_y = 0;
		top_bnd++;
	}
	else if (!top && !right && left && bot)
	{
		blk_x++;
		blk_y++;
		offset_x = 1;
		offset_y = 0;
		top_bnd++;
		right_bnd -= 1;
	}
	else if (!bot && !left && top && right)
	{
		blk_x++;
		blk_y++;
		offset_x = 0;
		offset_y = 1;
		left_bnd++;
		bot_bnd -= 1;
	}
	else if (!bot && !right && top && left)
	{
		blk_x++;
		blk_y++;
		offset_x = 1;
		offset_y = 1;
		right_bnd -= 1;
		bot_bnd -= 1;
	}
	else if (!bot && left && right && top)
	{
		blk_x +=2;
		blk_y++;
		offset_x = 1;
		offset_y = 1;
		bot_bnd -= 1;
	}
	else if (top && bot && !left && right)
	{
		blk_x++;
		blk_y +=2;
		offset_x = 0;
		offset_y = 1;
		left_bnd++;
	}
	else if (top && bot && !right && left)
	{
		blk_x++;
		blk_y +=2;
		offset_x = 1;
		offset_y = 1;
		right_bnd -= 1;
	}
	else if (top && bot && left && right)
	{
		blk_x +=2;
		blk_y +=2;
		offset_x = 1;
		offset_y = 1;
	}
	else if (!top && !bot && !left && !right)
	{
		offset_x = 0;
		offset_y = 0;
		top_bnd++;
		bot_bnd -= 1;
		left_bnd++;
		right_bnd -= 1;
	}
	else if (!top && !bot && !left && right)
	{
		blk_x++;
		offset_x = 0;
		offset_y = 0;
		top_bnd++;
		bot_bnd -= 1;
		left_bnd++;
	}
	else if (!top && !bot && left && !right)
	{
		blk_x++;
		offset_x = 1;
		offset_y = 0;
		top_bnd++;
		bot_bnd -= 1;
		right_bnd -= 1;
	}
	else if (!top && !bot && left && right)
	{
		blk_x += 2;
		offset_x = 1;
		offset_y = 0;
		top_bnd++;
		bot_bnd -= 1;
	}
	else if (!top && bot && !left && !right)
	{
		blk_y++;
		offset_x =0;
		offset_y = 0;
		top_bnd++;
		left_bnd++;
		right_bnd -= 1;
	}
	else if (top && !bot && !left && !right)
	{
		blk_y++;
		offset_x = 0;
		offset_y = 1;
		left_bnd++;
		right_bnd -= 1;
		bot_bnd -=1;
	}
	else if (top && bot && !left && !right)
	{
		blk_y += 2;
		offset_x = 0;
		offset_y = 1;
		left_bnd++;
		right_bnd -= 1;
	}
	
	u_old = (double*)malloc(blk_x*blk_y*sizeof(double));
	u = (double*)malloc(blk_x*blk_y*sizeof(double));
	u_new = (double*)malloc(blk_x*blk_y*sizeof(double));
	
	memset(u, 0, blk_x*blk_y*sizeof(double));
	memset(u_old, 0, blk_x*blk_y*sizeof(double));
	memset(u_new, 0, blk_x*blk_y*sizeof(double));
	
	/*Initialization*/
	for(int i = top_bnd; i <= bot_bnd; ++i) 
	{
		for(int j = left_bnd; j <= right_bnd; ++j) 
		{
			double x, y;
			if (remain_x > 0 && mycrds[1] < remain_x)
				x = (j+(blk_x_const+1)*mycrds[1])*dx;
			else
				x = (j+(blk_x_const+1)*remain_x+blk_x_const*(mycrds[1]-remain_x))*dx;
				
			if (remain_y > 0 && mycrds[0] < remain_y)
				y = (i+(blk_y_const+1)*mycrds[0])*dx;
			else
				y = (i+(blk_y_const+1)*remain_y+blk_y_const*(mycrds[0]-remain_y))*dx;
			
			//if (glob_rank == 0)
				//printf("P: %d, x: %f, y: %f, top_bnd: %d, bot_bnd: %d, left_bnd: %d, right_bnd: %d\n", glob_rank, x, y, top_bnd, bot_bnd, left_bnd, right_bnd);
			
			/* u0 */
			u[(i+offset_y)*blk_x+j+offset_x] = initialize(x, y, 0);
	
			/* u1 */
			u_new[(i+offset_y)*blk_x+j+offset_x] = initialize(x, y, dt);
		}
	}
	
#ifdef VERIFY
	double max_error = 0.0, glob_max_error= 0.0;
#endif

	if (remain_y > 0 && mycrds[0] < remain_y)
		MPI_Type_vector(blk_y_const+1, 1, blk_x, MPI_DOUBLE, &col);
	else
		MPI_Type_vector(blk_y_const, 1, blk_x, MPI_DOUBLE, &col);
	MPI_Type_commit(&col);
	
	if (remain_x > 0 && mycrds[1] < remain_x)
		MPI_Type_vector(1, blk_x_const+1, blk_x, MPI_DOUBLE, &row);
	else
		MPI_Type_vector(1, blk_x_const, blk_x, MPI_DOUBLE, &row);
	MPI_Type_commit(&row);
	
	for (k = 2; k < Nt; ++k)
	{
		if (right)
		{
			MPI_Isend(&u_new[(offset_y+1)*blk_x-2], 1, col, mycrds[1]+1, mycrds[1], \
						crt_row, &send_req[0]);
			
			MPI_Irecv(&u_new[(offset_y+1)*blk_x-1], 1, col, mycrds[1]+1, mycrds[1]+1, \
						crt_row, &recv_req[0]);
		}
		if (left)
		{
			MPI_Isend(&u_new[offset_y*blk_x+1], 1, col, mycrds[1]-1, mycrds[1], \
						crt_row, &send_req[1]);
			
			MPI_Irecv(&u_new[offset_y*blk_x], 1, col, mycrds[1]-1, mycrds[1]-1, \
						crt_row, &recv_req[1]);
		}
		if (top)
		{
			MPI_Isend(&u_new[blk_x+offset_x], 1, row, mycrds[0]-1, mycrds[0], \
						crt_col, &send_req[2]);
			
			MPI_Irecv(&u_new[offset_x], 1, row, mycrds[0]-1, mycrds[0]-1, \
						crt_col, &recv_req[2]);
		}	
		if (bot)
		{
			MPI_Isend(&u_new[blk_x*(blk_y-2)+offset_x], 1, row, mycrds[0]+1, mycrds[0], \
						crt_col, &send_req[3]);
			
			MPI_Irecv(&u_new[blk_x*(blk_y-1)+offset_x], 1, row, mycrds[0]+1, mycrds[0]+1, \
						crt_col, &recv_req[3]);
		}
		MPI_Barrier(crt_row);
		MPI_Barrier(crt_col);
		
		/*if (glob_rank==8 && k==2)
		{
			//printf("offset x: %d\n", offset_x);
			//printf("offset y: %d\n", offset_y);
		
			printf("Nt: %d\n", k);
			for(i=0; i<blk_y; ++i)
			{
				for(j=0; j<blk_x; ++j)
				{
					printf("%f ", u_new[i*blk_x+j]);
				}
				printf("\n");
			}
		}*/
		
		/* Swap ptrs */
		double *tmp = u_old;
		u_old = u;
		u = u_new;
		u_new = tmp;
	
		/* Apply stencil */
		for(i = 1; i < (blk_y-1); ++i)
		{
			for(j = 1; j < (blk_x-1); ++j)
			{
				
				u_new[i*blk_x+j] = 2*u[i*blk_x+j] - u_old[i*blk_x+j] + lambda2*
								(u[(i+1)*blk_x+j] + u[(i-1)*blk_x+j] + u[i*blk_x+j+1] \
								+ u[i*blk_x+j-1] - 4*u[i*blk_x+j]);
			}
		}

#ifdef VERIFY
		double error = 0.0;
		for(i = top_bnd; i <= bot_bnd; ++i) {
			for(j = left_bnd; j <= right_bnd; ++j) {
				double x, y, e;
				
				if (remain_x > 0 && mycrds[1] < remain_x)
				x = (j+(blk_x_const+1)*mycrds[1])*dx;
			else
				x = (j+(blk_x_const+1)*remain_x+blk_x_const*(mycrds[1]-remain_x))*dx;
				
			if (remain_y > 0 && mycrds[0] < remain_y)
				y = (i+(blk_y_const+1)*mycrds[0])*dx;
			else
				y = (i+(blk_y_const+1)*remain_y+blk_y_const*(mycrds[0]-remain_y))*dx;
				
				e = fabs(u_new[(i+offset_y)*blk_x+j+offset_x]- \
						initialize(x, y, k*dt));
				if(e>error)
					error = e;
			}
		}
		if(error > max_error)
			max_error = error;
#endif
		
	}

#ifdef VERIFY
	MPI_Reduce(&max_error, &glob_max_error, 1, MPI_DOUBLE, MPI_MAX, 0, com_crt);
	if (glob_rank == 0)
		printf("Maximum error: %g\n", glob_max_error);
#endif
	
	free(u);
	free(u_new);
	free(u_old);
	
	MPI_Finalize();
}

double timer()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	double seconds = tv.tv_sec + (double)tv.tv_usec / 1000000;
	return seconds;
}

double initialize(double x, double y, double t)
{
	double value = 0;
	#ifdef VERIFY
		/* standing wave */
		value = sin(3*M_PI*x)*sin(4*M_PI*y)*cos(5*M_PI*t);
	#else
	/* squared-cosine hump */
	const double width = 0.5;

	double centerx = 0.25;
	double centery = 0.5;

	double dist = sqrt((x-centerx)*(x-centerx) + (y-centery)*(y-centery));
	if(dist < width) {
		double cs = cos(M_PI_2*dist/width);
		value = cs*cs;
	}
#endif
  return value;
}
