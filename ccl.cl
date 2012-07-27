#define SCALE_FACTOR .0021
#define MIN_DISTANCE -10

void convert_screen_coords_to_mm (int width,
                                  int height,
                                  int dimension_reduction,
                                  int i,
                                  int j,
                                  int z,
                                  int *x,
                                  int *y)
{
  *x = round((i * dimension_reduction - width * dimension_reduction / 2.0) *
             (z + MIN_DISTANCE) * SCALE_FACTOR * (width / height));
  *y = round((j * dimension_reduction - height * dimension_reduction / 2.0) *
             (z + MIN_DISTANCE) * SCALE_FACTOR);
}

int get_distance(int a_x, int a_y, int a_z, int b_x, int b_y, int b_z)
{
  int dx, dy, dz;

  dx = abs (a_x - b_x);
  dy = abs (a_y - b_y);
  dz = abs (a_z - b_z);

  return sqrt (dx * dx + dy * dy + dz * dz);
}

__kernel void initialize_labels(__global unsigned int *labels, int size)
{
  int tid = get_global_id(0);

  if (tid < size)
    {
      labels[tid] = tid;
    }
}

__kernel void make_graph(__global unsigned short *buffer, __global int *mask_matrix, __global int *edge_matrix, __global int *weight_matrix, int width, int height)
{
  int i, j, tid, size;

  i = get_global_id(1);
  j = get_global_id(0);

  tid = j * width + i;
  size = width * height;

  if (tid < size)
    {
      for (int k=(i-1); k<=(i+1); k++)
        {
          for (int l=(j-1); l<=(j+1); j++)
            {
              if (k >= 0 && k < height && l>= 0 && l < width && (k != i || l != j))
                {
                  unsigned int neighbor = k * width + l;
                }
            }
        }
    }
}

__kernel void mesh_kernel(__global unsigned short *buffer, __global unsigned int *labels, __global int
    *mD, int width, int height)
{
  int id, idL, label, workgroup_size, i, j, block_start, index;
  int nId[8];
  int size;

  size = width * height;

  workgroup_size = get_local_size (1);

  block_start = workgroup_size * width * get_group_id (1) + get_group_id (0) *
    workgroup_size;

  id = block_start + get_local_id(1) * width + get_local_id(0);

  if (id < size)
    {
      i = get_global_id (1);
      j = get_global_id (0);

      index = 0;

      for (int k=(i-1); k<=(i+1); k++)
        {
          for (int l=(j-1); l<=(j+1); l++)
            {
              if (k >= 0 && k < height && l >= 0 && l < width && (k != i || l != j))
                {
                  unsigned int neighbor = k * width + l;

                  if ((buffer[id] == 0) && (buffer[neighbor] == 0))
                    {
                      nId[index] = neighbor;
                      index++;
                    }
                  else if (buffer[id] != 0 && buffer[neighbor] != 0)
                    {
                      nId[index] = neighbor;
                      index++;
                    }
                }
            }
        }

      label = labels[id];

      for (int i=0; i<index; i++)
        {
          if (labels[nId[i]] < label)
            {
              label = labels[nId[i]];
              // Should it be atomic?
              *mD = 1;
            }
        }
/*
      __local int local_mD;

      i = get_local_id (1);
      j = get_local_id (0);
      idL = get_local_id (1) * workgroup_size + get_local_id (0);

      for (int k=(i-1); k<=(i+1); k++)
        {
          for (int l=(j-1); l<=(j+1); l++)
            {
              if (k>=

            }
        }


      do
        {
          local_labels[idL] = label;

          barrier(CLK_LOCAL_MEM_FENCE);

          local_mD = 0;

          for (int i=0; i<index; i++)
            {
              if (local_labels[nId[i]] < label)
                {
                  label = local_labels[nId[i]];
                  local_mD = 1;
                }
            }

          barrier(CLK_LOCAL_MEM_FENCE);
        } while (local_mD);

     */ 
      labels[id] = label;
    }
}  
