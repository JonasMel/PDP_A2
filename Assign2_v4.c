

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);

int main(int argc, char **argv)
{
	if (argc != 4)
		return -1;
		
	
	
	int Nx, Ny, Nt, px, py, i, j, k, left, right, top, bot;
	double dt, dx, lambda2, *u_glob, *u, *u_old, *u_new, *u0;
	int blk_x, blk_y, blk_size, blk_y_const, blk_x_const, offset_x, offset_y;
	
	Nx = atoi(argv[1]);
	px = atoi(argv[2]);
	py = atoi(argv[3]);
	Ny = Nx;
	Nt = Nx;
	
	if (Nx % px == 0)
	{
		blk_x = Nx/px;
	}
	
	if (Ny % py == 0)
	{
		blk_y = Ny/py;
	}
	
	blk_x_const = blk_x;
	blk_y_const = blk_y;
	blk_size = blk_x*blk_y;
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
	MPI_Datatype blk, row, col;
	MPI_Type_vector(blk_size, blk_x, Nx, MPI_DOUBLE, &blk);
	MPI_Type_commit(&blk);

	MPI_Type_vector(blk_y, 1, Nx, MPI_DOUBLE, &col);
	MPI_Type_commit(&col);
	MPI_Type_vector(1, blk_x, Nx, MPI_DOUBLE, &row);
	MPI_Type_commit(&row);

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

	if (px*py > 1)
	{
		if (!top && !left)
		{
			blk_x++;
			blk_y++;
			offset_x = 0;
			offset_y = 0;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (!top && left && right)
		{
			blk_x +=2;
			blk_y++;
			offset_x = 1;
			offset_y = 0;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (!top && !right)
		{
			blk_x++;
			blk_y++;
			offset_x = 1;
			offset_y = 0;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (!bot && !left)
		{
			blk_x++;
			blk_y++;
			offset_x = 0;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (!bot && !right)
		{
			blk_x++;
			blk_y++;
			offset_x = 1;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (!bot && left && right)
		{
			blk_x +=2;
			blk_y++;
			offset_x = 1;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (top && bot && !left)
		{
			blk_x++;
			blk_y +=2;
			offset_x = 0;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (top && bot && !right)
		{
			blk_x++;
			blk_y +=2;
			offset_x = 1;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
		else if (top && bot && left && right)
		{
			blk_x +=2;
			blk_y +=2;
			offset_x = 1;
			offset_y = 1;
			u_old = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
			u_new = (double*)malloc((blk_x)*(blk_y)*sizeof(double));
		}
	}
	else
	{
		u_old = (double*)malloc(blk_x*blk_y*sizeof(double));
		u = (double*)malloc(blk_x*blk_y*sizeof(double));
		u_new = (double*)malloc(blk_x*blk_y*sizeof(double));
	}
	
	/*Initialization*/
	for(int i = 1; i < (blk_y-1); ++i) 
	{
		for(int j = 1; j < (blk_x-1); ++j) 
		{
			double x = (j)*blk_x_const*mycrds[1]*dx;
			double y = (i)*blk_y_const*mycrds[0]*dx;

			/* u0 */
			u[i*(blk_x +offset_y)+j+offset_x] = initialize(x, y, 0);
	
			/* u1 */
			u_new[i*(blk_x +offset_y)+j+offset_x] = initialize(x, y, dt);
		}
	}
	for (k = 0; k < 25; ++k)
	{

		
		if (right)
		{
			MPI_Isend(&u_new[blk_x-2], 1, col, mycrds[1]+1, mycrds[1], \
						crt_row, &send_req[0]);
			
			MPI_Irecv(&u_new[blk_x-1], 1, col, mycrds[1]+1, mycrds[1]+1, \
						crt_row, &recv_req[2]);
		}
		if (left)
		{
			MPI_Isend(&u_new[1], 1, col, mycrds[1]-1, mycrds[1], \
						crt_row, &send_req[2]);
			
			MPI_Irecv(&u_new[0], 1, col, mycrds[1]-1, mycrds[1]-1, \
						crt_row, &recv_req[0]);
		}
		if (top)
		{
			MPI_Isend(&u_new[blk_x], 1, row, mycrds[0]-1, mycrds[0], \
						crt_col, &send_req[3]);
			
			MPI_Irecv(&u_new[0], 1, row, mycrds[0]-1, mycrds[0]-1, \
						crt_col, &recv_req[1]);
		}	
		if (bot)
		{
			MPI_Isend(&u_new[blk_x*(blk_y-2)], 1, row, mycrds[0]+1, mycrds[0], \
						crt_col, &send_req[3]);
			
			MPI_Irecv(&u_new[blk_x*(blk_y-1)], 1, row, mycrds[0]+1, mycrds[0]+1, \
						crt_col, &recv_req[1]);
		}
		MPI_Barrier(crt_row);
		MPI_Barrier(crt_col);
		
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
		
	}
	
	free(u);
	free(u_new);
	free(u_old);
	
	if (glob_rank == 0)
	{
		free(u0);
		free(u_glob);
	}
	
	MPI_Finalize();
	
	
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