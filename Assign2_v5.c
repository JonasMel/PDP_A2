#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

#define VERIFY 1

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);

int main(int argc, char **argv)
{
	if (argc != 4)
		return -1;
		
	int Nx, Ny, Nt, px, py, i, ii, j, k, left, right, top, bot, remainx, remainy;
	double *u_glob, *u, *u_old, *u_new;
	double dt, dx, lambda2, begin, end, max_error = 0.0;
	int blk_x, blk_y, blk_size, blk_y_const, blk_x_const, offset_x, offset_y;
	
	Nx = atoi(argv[1]);
	px = atoi(argv[2]);
	py = atoi(argv[3]);
	Ny = Nx;
	Nt = Nx;

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
	
	/*Fixing block sizes and starting position if Nx or Ny is not divided evenly.*/
	remainx = Nx % px;
	blk_x = (int)Nx/px;
	blk_x_const = blk_x;		
	int left_bnd = 0;
	int right_bnd;;
	if ( remainx > 0 && mycrds[1] < remainx)
	{
		blk_x++;
		right_bnd = blk_x_const;
	}
	else
		right_bnd = blk_x_const-1;
		
	remainy = Ny % py;
	blk_y = (int)Ny/py;
	blk_y_const = blk_y;	
	int top_bnd = 0;
	int bot_bnd;
	if ( remainy > 0 && mycrds[0] < remainy)
	{
		blk_y++;
		bot_bnd = blk_y_const;
	}
	else
		bot_bnd = blk_y_const-1;
			
	int startx, starty;
	if ( remainx > mycrds[1] && mycrds[1] > 0)
	{
		startx = blk_x*mycrds[1];
	}
	else if (mycrds[1] >= remainx && remainx > 0)
	{
		startx = blk_x*mycrds[1] + remainx;
	}
	else 
	{
		startx = blk_x_const*mycrds[1];
	}
	
	
	if (remainy > mycrds[0] && mycrds[0] > 0)
	{
		starty = blk_y*mycrds[0];
	}
	else if (mycrds[0] >= remainy && remainy > 0)
	{
		starty = blk_y*mycrds[0] + remainy;
	}
	else
	{
		starty = blk_y_const*mycrds[0];
	}


	
	
	blk_size = blk_x*blk_y;
	dx = 1.0 / (Nx-1);
	dt = 0.5*dx;
	lambda2 = (dt/dx)*(dt/dx);
	
	
	/*Creating vector types for communication*/
	MPI_Datatype blk, row, col;
	MPI_Type_vector(blk_y, blk_x, Nx, MPI_DOUBLE, &blk);
	MPI_Type_commit(&blk);

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
			double x = (j+startx)*dx;
			double y = (i+starty)*dx;

			/* u0 */
			u[(i+offset_y)*blk_x+j+offset_x] = initialize(x, y, 0);
	
			/* u1 */
			u_new[(i+offset_y)*blk_x+j+offset_x] = initialize(x, y, dt);
		}
	}
/*	for(i=0; i<blk_y; ++i)
	{
		for(j=0; j<blk_x; ++j)
		{
			printf("%f ", u_new[i*blk_x+j]);
		}
		printf("\n");
	}
*/
		/*	if (crt_rank == 0)
			{	
				printf("Process coords: (%d, %d)\n", mycrds[0], mycrds[1]);
				for (i = top_bnd; i <= bot_bnd; ++i)
				{
					for (j = left_bnd; j <= right_bnd; ++j)
					{
						printf(" %lf", u_new[i*blk_y + j]);
					}
					printf("\n");
				}
			}*/
	if (remainy && mycrds[0] < remainy)
		MPI_Type_vector(blk_y_const+1, 1, blk_x, MPI_DOUBLE, &col);
	else
		MPI_Type_vector(blk_y_const, 1, blk_x, MPI_DOUBLE, &col);
		
	MPI_Type_commit(&col);
	if (remainx && mycrds[1] < remainx)
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
		
		//for (ii = 0; ii < px*py; ++ii)
		//{
	/*		if (crt_rank == 0)
			{	
				printf("Process coords: (%d, %d)\n", mycrds[0], mycrds[1]);
				for (i = 0; i < blk_y; ++i)
				{
					for (j = 0; j < blk_x; ++j)
					{
						printf(" %lf", u_new[i*blk_y + j]);
					}
					printf("\n");
				}
			}*/
		//}

 		
		/*if (glob_rank==15)
		{
			//printf("offset x: %d\n", offset_x);
			//printf("offset y: %d\n", offset_y);
		
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
				
				u_new[i*blk_x+j] = 2*u[i*blk_x+j] - u_old[i*blk_x+j] + lambda2* \
								(u[(i+1)*blk_x+j] + u[(i-1)*blk_x+j] + u[i*blk_x+j+1] \
								+ u[i*blk_x+j-1] - 4*u[i*blk_x+j]);
			}
		}
#ifdef VERIFY	
		double error=0.0;
	//	if (crt_rank == 0) printf("crt_rank is %d\n", crt_rank);
		for(int i = top_bnd; i <= bot_bnd; ++i) 
		{
			for(int j = left_bnd; j <= right_bnd; ++j)
			{
				double e = fabs(u_new[(i+offset_y)*blk_x+j+offset_x]-initialize((j+startx)*dx, \
								(i+starty)*dx, k*dt));
		/*		if (crt_rank == 0)
				{
				printf(" %lf", initialize((j+blk_x_const*mycrds[1])*dx, \
								(i+blk_y_const*mycrds[0])*dx, k*dt));
				}*/
				if(e>error)
					error = e;
			}
		/*	if (crt_rank == 0)
				printf("\n");*/
		}
#endif
		/*if (crt_rank == 0)
		{	
			printf("Process coords: (%d, %d)\n", mycrds[0], mycrds[1]);
			for (i = top_bnd; i <= bot_bnd; ++i)
			{
				for (j = left_bnd; j <= right_bnd; ++j)
				{
					printf(" %lf", u_new[(i+offset_y)*blk_x+j+offset_x]);
				}
				if (crt_rank == 0)
					printf("\n");
			}
		}	*/
   if(error > max_error)
      max_error=error;
		
	
	}
	printf("max error = %4.16lf\n", max_error);
	
	free(u);
	free(u_new);
	free(u_old);
	
/*	if (glob_rank == 0)
		free(u_glob);
	*/
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
  const double width = 0.1;

  double centerx = 0.25;
  double centery = 0.5;

  double dist = sqrt((x-centerx)*(x-centerx) +
                     (y-centery)*(y-centery));
  if(dist < width) {
    double cs = cos(M_PI_2*dist/width);
    value = cs*cs;
  }
#endif
  return value;
}

void save_solution(double *u, int Ny, int Nx, int n)
{
  char fname[50];
  sprintf(fname, "solution-%d.dat", n);
  FILE *fp = fopen(fname, "w");

  fprintf(fp, "%d %d\n", Nx, Ny);

  for(int j = 0; j < Ny; ++j) {
    for(int k = 0; k < Nx; ++k) {
      fprintf(fp, "%e\n", u[j*Nx+k]);
    }
  }

  fclose(fp);
}