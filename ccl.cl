__kernel void initialize_labels(__global int *labels, int size)
{
  int tid = get_global_id(0);

  printf("tid = %d\n", tid);

  if (tid < size)
    {
      labels[tid] = tid;
    }
}

__kernel void mesh_kernel(__global int *buffer, __global int *labels, __global int
    *mD, int width, int height, __local int *local_labels)
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

      for (int k=-1; k<=1; k++)
        {
          for (int l=-1; l<=1; l++)
            {
              int element = labels[j * width + i];

              if (labels[id] == 0 && element == 0)
                {
                  nId[index] = element;
                  index++;
                }
              else if (labels[id] != 0 && element != 0)
                {
                  nId[index] = element;
                  index++;
                }
            }
        }

      label = labels[id];

      for (int i=0; i<index; i++)
        {
          if (labels[nId[i]] < label)
            {
              label = labels[nId[i]];
              *mD = 1;
            }
        }

      __local int local_mD;

      idL = get_local_id (1) * workgroup_size + get_local_id (0);
      local_mD = 1;

      index = 0;

      for (int k=-1; k<=1; k++)
        {
          for (int l=-1; l<=1; l++)
            {
              int element = labels[j * width + i];

              if (labels[id] == 0 && element == 0)
                {
                  nId[index] = element;
                  index++;
                }
              else if (labels[id] != 0 && element != 0)
                {
                  nId[index] = element;
                  index++;
                }
            }
        }

      while (local_mD)
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
        }

      labels[id] = label;
    }
}  
