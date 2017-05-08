/* -*- c-basic-offset:2; tab-width:2; indent-tabs-mode:nil -*-
 *
 */

// mpicc -O3 -o wave wave.c -lmpi -lm
// mpirun -np 32 ./wave 128 8 4

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <sys/time.h>
#include <mpi.h>

//#define WRITE_TO_FILE
#define VERIFY

double timer();
double initialize(double x, double y, double t);
void save_solution(double *u, int Ny, int Nx, int n);

int main(int argc, char *argv[])
{
  if (argc != 4)
	{
		printf("To run the program you need to input Nx=Ny: Number of elements, Px: Processors in x and Py: Processors in y\n");
		return -1;
	}
	
  int Nx, Ny, Nt, px, py;
  double dt, dx, lambda_sq;
  double *u_new_all;
  double *u;
  double *u_old;
  double *u_new;
  double begin, end;

  if(argc>1) {
    Nx = atoi(argv[1]);
    px = atoi(argv[2]);
    py = atoi(argv[3]);
  }
  
  Ny = Nx;
  Nt = Nx;
  
  int blk_x_static, blk_y_static;

  int blk_x = Nx/px;
  int blk_y = Ny/py;
  int blk_size = blk_x*blk_y;
  blk_x_static = blk_x;
  blk_y_static = blk_y;
  dx = 1.0/(Nx-1);
  dt = 0.50*dx;
  lambda_sq = (dt/dx)*(dt/dx);
  
  MPI_Init(&argc, &argv);
  int my_world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &my_world_rank);  
  
  int ndims = 2;
  int dims[] = {py, px};
  int cyclic[] = {0, 0};
  int reorder = 1;
  int my_cart_rank;
  int my_coords[ndims];
  MPI_Comm comm_cart;
  MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, cyclic, reorder, &comm_cart);
  MPI_Comm_rank(comm_cart, &my_cart_rank);
  MPI_Cart_coords(comm_cart, my_cart_rank, ndims, my_coords);
  
  if (my_cart_rank == 0)
    u_new_all = malloc(Nx*Ny*sizeof(double));

  if ((my_coords[1] == 0 || my_coords[1] == px-1) && (my_coords[0] == 0 || my_coords[0] == py-1)) { // corner points
    blk_x++;
    blk_y++;
  } else if (my_coords[1] == 0 || my_coords[1] == px-1) { // edges columns
    blk_x++;
    blk_y += 2;
  } else if (my_coords[0] == 0 || my_coords[0] == py-1) { // edges rows
    blk_x += 2;
    blk_y++;
  } else { // inner points
    blk_x += 2;
    blk_y += 2;
  }
  
  u = malloc(blk_x*blk_y*sizeof(double));
  u_old = malloc(blk_x*blk_y*sizeof(double));
  u_new = malloc(blk_x*blk_y*sizeof(double));
  
  /* Setup IC */
  memset(u, 0, blk_x*blk_y*sizeof(double));
  memset(u_old, 0, blk_x*blk_y*sizeof(double));
  memset(u_new, 0, blk_x*blk_y*sizeof(double));

  for(int i = 1; i < (blk_y-1); ++i) {
    for(int j = 1; j < (blk_x-1); ++j) {
      double x = (j+1)*blk_x*my_coords[1]*dx;
      double y = (i+1)*blk_y*my_coords[0]*dx;

      /* u0 */
      u[i*blk_x+j] = initialize(x, y, 0);

      /* u1 */
      u_new[i*blk_x+j] = initialize(x, y, dt);
    }
  }
  
  /* Collect blocks */
  MPI_Datatype blk_vector;
  MPI_Type_vector(blk_y_static, blk_x_static, blk_x, MPI_DOUBLE, &blk_vector);
  MPI_Type_commit(&blk_vector);
  int offset;
  
  // Send blocks to root node
  if (my_coords[1] == 0 && my_coords[0] == 0) // upper left corner
    offset = 0;
  else if (my_coords[0] == 0) // upper edge row
    offset = 1;
  else if (my_coords[1] == 0) // left edge column
    offset = blk_x;
  else // all other cases
    offset = blk_x+1;

  MPI_Request send_request[px*py];
  int blk_size_static;
  blk_size_static = blk_x_static*blk_y_static*sizeof(MPI_DOUBLE);
  MPI_Isend(&u_new[offset], 1, blk_vector, 0, my_cart_rank, comm_cart, &send_request[my_cart_rank]);
  printf("%d sent! Block size is %d\n", my_cart_rank, blk_size_static);
  
  // Receive blocks
  int source_coords[ndims];
  MPI_Request recv_request[px*py];
  MPI_Status status[px*py];
  if (my_cart_rank == 0)
  {
    for (int p=0; p<px*py; ++p)
    {
      printf("probing %d...\n", p);
      MPI_Probe(p, p, comm_cart, status);
      MPI_Cart_coords(comm_cart, p, ndims, source_coords);
      MPI_Recv(&u_new_all[source_coords[0]*Nx*blk_y_static \
		+ source_coords[1]*blk_x_static], 1, blk_vector, p, p, comm_cart, status);
      printf("%d received!\n", p);
    }
  }
  
  //return 0;

#ifdef WRITE_TO_FILE
  save_solution(u_new_all, Ny, Nx, 1);
#endif
#ifdef VERIFY
  double max_error = 0.0;
#endif

  int next_coords[ndims], prev_coords[ndims];
  int next_cart_rank, prev_cart_rank;

  /* Integrate */

  begin = timer();
  for(int n=2; n<Nt; ++n) {
    
    /* Cols communication */
    MPI_Datatype col_vector;
    MPI_Type_vector(blk_y, 1, blk_x, MPI_DOUBLE, &col_vector);
    MPI_Type_commit(&col_vector);
    
    next_coords[1] = my_coords[1]+1;
    next_coords[0] = my_coords[0];
    prev_coords[1] = my_coords[1]-1;
    prev_coords[0] = my_coords[0];
    if (next_coords[1]>=0 && next_coords[1]<px && next_coords[0]>=0 \
		&& next_coords[0]<py)
      MPI_Cart_rank(comm_cart, next_coords, &next_cart_rank);
    if (prev_coords[1]>=0 && prev_coords[1]<px && prev_coords[0]>=0 \
		&& prev_coords[0]<py)
      MPI_Cart_rank(comm_cart, prev_coords, &prev_cart_rank);
    
    if (my_coords[1] == 0) { // left edge nodes
      MPI_Send(&u[blk_x-2], 1, col_vector, next_cart_rank, next_cart_rank, comm_cart);
    } else if (my_coords[1] != 0 && my_coords[1] != px-1) { // inner nodes
      MPI_Sendrecv(&u[blk_x-2], 1, col_vector, next_cart_rank, next_cart_rank, &u[0], \
			1, col_vector, prev_cart_rank, my_cart_rank, comm_cart, MPI_STATUS_IGNORE);
    } else if (my_coords[1] == px-1) { // right edge nodes
      MPI_Recv(&u[0], 1, col_vector, prev_cart_rank, my_cart_rank, comm_cart, MPI_STATUS_IGNORE);
    }
    
    /* Rows communication */  
    next_coords[1] = my_coords[1];
    next_coords[0] = my_coords[0]+1;
    prev_coords[1] = my_coords[1];
    prev_coords[0] = my_coords[0]-1;
    if (next_coords[1]>=0 && next_coords[1]<px \
		&& next_coords[0]>=0 && next_coords[0]<py)
      MPI_Cart_rank(comm_cart, next_coords, &next_cart_rank);
    if (prev_coords[1]>=0 && prev_coords[1]<px \
		&& prev_coords[0]>=0 && prev_coords[0]<py)
      MPI_Cart_rank(comm_cart, prev_coords, &prev_cart_rank);
    
    if (my_coords[0] == 0) { // top edge nodes
      MPI_Send(&u[blk_x-2], 1, MPI_DOUBLE, next_cart_rank, next_cart_rank, comm_cart);
    } else if (my_coords[1] != 0 && my_coords[1] != px-1) { // inner nodes
      MPI_Sendrecv(&u[blk_x-2], 1, col_vector, next_cart_rank, next_cart_rank, &u[0], \
			1, col_vector, prev_cart_rank, my_cart_rank, comm_cart, MPI_STATUS_IGNORE);
    } else if (my_coords[1] == px-1) { // botton edge nodes
      MPI_Recv(&u[0], 1, col_vector, prev_cart_rank, my_cart_rank, comm_cart, MPI_STATUS_IGNORE);
    }
  
    /* Swap ptrs */
    double *tmp = u_old;
    u_old = u;
    u = u_new;
    u_new = tmp;

    /* Apply stencil */
    for(int i = 1; i < (blk_y-1); ++i) {
      for(int j = 1; j < (blk_x-1); ++j) {

        u_new[i*blk_x+j] = 2*u[i*blk_x+j] - u_old[i*blk_x+j] + lambda_sq*
          (u[(i+1)*blk_x+j] + u[(i-1)*blk_x+j] + u[i*blk_x+j+1] + u[i*blk_x+j-1] - 4*u[i*blk_x+j]);
      }
    }

#ifdef VERIFY
    double error = 0.0;
    for(int i = 0; i < Ny; ++i) {
      for(int j = 0; j < Nx; ++j) {
        double e = fabs(u_new[i*Nx+j] - initialize(j*dx,i*dx,n*dt));
        if(e>error)
          error = e;
      }
    }
    if(error > max_error)
      max_error = error;
#endif

  // Send blocks to root node
  

    MPI_Isend(&u_new[offset], 1, blk_vector, 0, my_cart_rank,\
				comm_cart, &send_request[my_cart_rank]);
  
  // Receive blocks
  int source_coords[ndims];
  if (my_cart_rank == 0)
  {
    for (int p=0; p<px*py; ++p)
    {
	  MPI_Probe(p, p, comm_cart, status);
      MPI_Cart_coords(comm_cart, p, ndims, source_coords);
      MPI_Recv(&u_new_all[source_coords[0]*Nx*blk_y_static + source_coords[1]*blk_x_static], \
		1, blk_vector, p, p, comm_cart, MPI_STATUS_IGNORE);
		  printf("%d recieved second time\n", p);
    }
  }

#ifdef WRITE_TO_FILE
    save_solution(u_new, Ny, Nx, n);
#endif

  }
  end = timer();

  printf("Time elapsed: %g s\n", (end-begin));

#ifdef VERIFY
  printf("Maximum error: %g\n", max_error);
#endif

  if (my_cart_rank == 0)
	free(u_new_all);
  free(u);
  free(u_old);
  free(u_new);
  
  MPI_Finalize();

  return 0;
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
