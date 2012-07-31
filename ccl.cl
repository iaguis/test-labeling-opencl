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

__kernel void initialize(__global unsigned int *labels, __global int *edge_matrix, __global int *weight_matrix, int size)
{
  int tid = get_global_id(0);

  if (tid < size)
    {
      labels[tid] = tid;
      for (int i=0; i<8; i++)
        {
          edge_matrix[(tid*8)+i] = -1;
          weight_matrix[(tid*8)+i] = -1;
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
                  else 
                  if ((buffer[id] != 0) && (buffer[neighbor] != 0))
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
          if (buffer[nId[i]] == 0)
            {
              labels[nId[i]] = 0;
            }
          if (labels[nId[i]] < label)
            {
              label = labels[nId[i]];
              // Should it be atomic?
              atomic_inc(mD);
            }
        }
      labels[id] = label;
    }
}

__kernel void
make_graph (__global unsigned short *buffer,
            __global unsigned int *labels,
            __global int *edge_matrix,
            __global int *weight_matrix,
            int width,
            int height,
            int label)
{
  int i, j, node, size, index;

  size = width * height;

  i = get_global_id (1);
  j = get_global_id (0);

  node = i * width + j;

  if (node < size)
    {
      index = 0;

      if (labels[node] != 0)
        {
        for (int k=(i-1); k<=(i+1); k++)
          {
            for (int l=(j-1); l<=(j+1); l++)
              {
                if (k >= 0 && k < height && l >= 0 && l < width && (k != i || l != j))
                  {
                    unsigned int neighbor = k * width + l;

                    if (labels[neighbor] == label)
                      {
                        int x_node, y_node, z_node, x_neigh, y_neigh, z_neigh;

                        z_node = buffer[node];
                        z_neigh = buffer[neighbor];

                        convert_screen_coords_to_mm(width, height, 16, i, j, z_node, &x_node, &y_node);
                        convert_screen_coords_to_mm(width, height, 16, l, k, z_neigh, &x_neigh, &y_neigh);

                        edge_matrix[node*8 + index] = neighbor;
                        weight_matrix[node*8 + index] = get_distance(x_node, y_node, z_node, x_neigh, y_neigh, z_neigh);
                        index++;
                      }
                  }
              }
           }
        }
    }
}
