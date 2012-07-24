#include <stdlib.h>
#include <stdio.h>
#include <glib.h>
#include <glib-object.h>
#include <gio/gio.h>

#include <clutter/clutter.h>

#include <CL/cl.h>

#define WIDTH  640
#define HEIGHT 480

typedef struct {
  /* Host data */
  gint *buffer_matrix;
  gint *labels_matrix;
  gint *mD;

  cl_platform_id platform;
  cl_device_id device;
  cl_context context;
  cl_command_queue command_queue;
  cl_program program;

  cl_mem buffer_matrix_device;
  cl_mem labels_matrix_device;
  cl_mem mD_device;

  cl_kernel initialize_labels;
  cl_kernel mesh_kernel;
} oclCclData;

void
check_error_file_line (int err_num,
                       int expected,
                       const char* file,
                       const int line_number);

#define check_error(a, b) check_error_file_line (a, b, __FILE__, __LINE__)

char *
error_desc (int err_num)
{
  switch (err_num)
    {
      case CL_SUCCESS:
        return "CL_SUCCESS";
        break;

      case CL_DEVICE_NOT_FOUND:
        return "CL_DEVICE_NOT_FOUND";
        break;

      case CL_DEVICE_NOT_AVAILABLE:
        return "CL_DEVICE_NOT_AVAILABLE";
        break;

      case CL_COMPILER_NOT_AVAILABLE:
        return "CL_COMPILER_NOT_AVAILABLE";
        break;

      case CL_MEM_OBJECT_ALLOCATION_FAILURE:
        return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        break;

      case CL_OUT_OF_RESOURCES:
        return "CL_OUT_OF_RESOURCES";
        break;

      case CL_OUT_OF_HOST_MEMORY:
        return "CL_OUT_OF_HOST_MEMORY";
        break;

      case CL_PROFILING_INFO_NOT_AVAILABLE:
        return "CL_PROFILING_INFO_NOT_AVAILABLE";
        break;

      case CL_MEM_COPY_OVERLAP:
        return "CL_MEM_COPY_OVERLAP";
        break;

      case CL_IMAGE_FORMAT_MISMATCH:
        return "CL_IMAGE_FORMAT_MISMATCH";
        break;

      case CL_IMAGE_FORMAT_NOT_SUPPORTED:
        return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        break;

      case CL_BUILD_PROGRAM_FAILURE:
        return "CL_BUILD_PROGRAM_FAILURE";
        break;

      case CL_MAP_FAILURE:
        return "CL_MAP_FAILURE";
        break;

      case CL_INVALID_VALUE:
        return "CL_INVALID_VALUE";
        break;

      case CL_INVALID_DEVICE_TYPE:
        return "CL_INVALID_DEVICE_TYPE";
        break;

      case CL_INVALID_PLATFORM:
        return "CL_INVALID_PLATFORM";
        break;

      case CL_INVALID_DEVICE:
        return "CL_INVALID_DEVICE";
        break;

      case CL_INVALID_CONTEXT:
        return "CL_INVALID_CONTEXT";
        break;

      case CL_INVALID_QUEUE_PROPERTIES:
        return "CL_INVALID_QUEUE_PROPERTIES";
        break;

      case CL_INVALID_COMMAND_QUEUE:
        return "CL_INVALID_COMMAND_QUEUE";
        break;

      case CL_INVALID_HOST_PTR:
        return "CL_INVALID_HOST_PTR";
        break;

      case CL_INVALID_MEM_OBJECT:
        return "CL_INVALID_MEM_OBJECT";
        break;

      case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
        return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        break;

      case CL_INVALID_IMAGE_SIZE:
        return "CL_INVALID_IMAGE_SIZE";
        break;

      case CL_INVALID_SAMPLER:
        return "CL_INVALID_SAMPLER";
        break;

      case CL_INVALID_BINARY:
        return "CL_INVALID_BINARY";
        break;

      case CL_INVALID_BUILD_OPTIONS:
        return "CL_INVALID_BUILD_OPTIONS";
        break;

      case CL_INVALID_PROGRAM:
        return "CL_INVALID_PROGRAM";
        break;

      case CL_INVALID_PROGRAM_EXECUTABLE:
        return "CL_INVALID_PROGRAM_EXECUTABLE";
        break;

      case CL_INVALID_KERNEL_NAME:
        return "CL_INVALID_KERNEL_NAME";
        break;

      case CL_INVALID_KERNEL_DEFINITION:
        return "CL_INVALID_KERNEL_DEFINITION";
        break;

      case CL_INVALID_KERNEL:
        return "CL_INVALID_KERNEL";
        break;

      case CL_INVALID_ARG_INDEX:
        return "CL_INVALID_ARG_INDEX";
        break;

      case CL_INVALID_ARG_VALUE:
        return "CL_INVALID_ARG_VALUE";
        break;

      case CL_INVALID_ARG_SIZE:
        return "CL_INVALID_ARG_SIZE";
        break;

      case CL_INVALID_KERNEL_ARGS:
        return "CL_INVALID_KERNEL_ARGS";
        break;

      case CL_INVALID_WORK_DIMENSION:
        return "CL_INVALID_WORK_DIMENSION";
        break;

      case CL_INVALID_WORK_GROUP_SIZE:
        return "CL_INVALID_WORK_GROUP_SIZE";
        break;

      case CL_INVALID_WORK_ITEM_SIZE:
        return "CL_INVALID_WORK_ITEM_SIZE";
        break;

      case CL_INVALID_GLOBAL_OFFSET:
        return "CL_INVALID_GLOBAL_OFFSET";
        break;

      case CL_INVALID_EVENT_WAIT_LIST:
        return "CL_INVALID_EVENT_WAIT_LIST";
        break;

      case CL_INVALID_EVENT:
        return "CL_INVALID_EVENT";
        break;

      case CL_INVALID_OPERATION:
        return "CL_INVALID_OPERATION";
        break;

      case CL_INVALID_GL_OBJECT:
        return "CL_INVALID_GL_OBJECT";
        break;

      case CL_INVALID_BUFFER_SIZE:
        return "CL_INVALID_BUFFER_SIZE";
        break;

      case CL_INVALID_MIP_LEVEL:
        return "CL_INVALID_MIP_LEVEL";
        break;

      case CL_INVALID_GLOBAL_WORK_SIZE:
        return "CL_INVALID_GLOBAL_WORK_SIZE";
        break;
    }
  return "";
}

void
check_error_file_line (int err_num,
                       int expected,
                       const char* file,
                       const int line_number)
{
  if (err_num != expected)
    {
      fprintf (stderr, "Line %d in File %s:", line_number, file);
      fprintf (stderr, "%s\n", error_desc (err_num));
      exit (1);
    }
}

cl_program
load_and_build_program (cl_context context,
                        cl_device_id device,
                        char *file_name)
{
  cl_program program;
  int program_size;
  cl_int err_num;

  FILE *program_f;

  char *program_buffer;

  program_f = fopen (file_name, "r");
  if (program_f == NULL)
    {
      fprintf (stderr, "%s not found\n", file_name);
      exit (1);
    }

  fseek (program_f, 0, SEEK_END);

  program_size = ftell (program_f);
  rewind (program_f);

  program_buffer = malloc (program_size + 1);
  program_buffer[program_size] = '\0';

  fread (program_buffer, sizeof (char), program_size, program_f);
  fclose (program_f);

  program = clCreateProgramWithSource (context, 1, (const char **)
                                       &program_buffer, NULL, &err_num);

  check_error (err_num, CL_SUCCESS);

  err_num = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
  if (err_num != CL_SUCCESS)
   {

     size_t size;

     err_num = clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
                                      check_error (err_num, CL_SUCCESS);

     char *log = malloc (size+1);

     err_num = clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG,
         size, log, NULL);

     fprintf (stderr, "%s\n", log);

     check_error (err_num, CL_SUCCESS);
   }

  return program;
}

cl_int
ocl_set_up_context (cl_device_type device_type,
                    cl_platform_id *platform,
                    cl_context *context,
                    cl_device_id *device,
                    cl_command_queue *command_queue)
{
  cl_int err_num;

  cl_context_properties contextProperties[] =
    {
      CL_CONTEXT_PLATFORM,
      (cl_context_properties)*platform,
      0
    };

  *context = clCreateContextFromType (contextProperties, device_type,
      NULL, NULL, &err_num);

  if (err_num != CL_SUCCESS)
    {
      /* FIXME add rest of the devices */
      if (device_type == CL_DEVICE_TYPE_CPU)
        printf ("No CPU devices found.\n");
      else if (device_type == CL_DEVICE_TYPE_GPU)
        printf ("No GPU devices found.\n");
    }

  err_num = clGetDeviceIDs (*platform, device_type, 1, device, NULL);

  if (err_num != CL_SUCCESS)
    {
      return err_num;
    }

  *command_queue = clCreateCommandQueue (*context, *device, 0, &err_num);

  if (err_num != CL_SUCCESS)
    {
      return err_num;
    }
  return 0;
}

static guint16 *
read_file_to_buffer (const gchar *name, gsize count, GError *e)
{
  GError *error = NULL;
  guint16 *depth = NULL;
  GFile *new_file = g_file_new_for_path (name);
  GFileInputStream *input_stream = g_file_read (new_file,
                                                NULL,
                                                &error);
  if (error != NULL)
    {
      g_debug ("ERROR: %s", error->message);
    }
  else
    {
      gsize bread = 0;
      depth = g_slice_alloc (count);
      g_input_stream_read_all ((GInputStream *) input_stream,
                               depth,
                               count,
                               &bread,
                               NULL,
                               &error);

      if (error != NULL)
        {
          g_debug ("ERROR: %s", error->message);
        }
    }
  return depth;
}

static guint16 *
reduce_depth_file (const gchar *name,
                   guint reduce_factor,
                   guint *reduced_width,
                   guint *reduced_height)
{
  guint i, j, r_width, r_height;
  guint16 *depth, *reduced_depth;
  GError *error = NULL;
  gsize count = WIDTH * HEIGHT * sizeof (guint16);

  depth = read_file_to_buffer (name, count, error);

  if (depth == NULL)
    return NULL;

  r_width = (WIDTH - WIDTH % reduce_factor) / reduce_factor;
  r_height = (HEIGHT - HEIGHT % reduce_factor) / reduce_factor;
  reduced_depth = g_slice_alloc (r_width * r_height * sizeof (guint16));

  for (i = 0; i < r_width; i++)
    {
      for (j = 0; j < r_height; j++)
        {
          guint index = j * WIDTH * reduce_factor + i * reduce_factor;
          reduced_depth[j * r_width + i] = depth[index];
        }
    }
  *reduced_width = r_width;
  *reduced_height = r_height;

  g_slice_free1 (count, depth);
  return reduced_depth;
}

void
ocl_init (oclCclData *data,
          gint matrix_size)
{
  if (data->platform == NULL)
    {
      cl_uint num_platforms;
      cl_int err_num;

      /* Find first OpenCL platform */
      err_num = clGetPlatformIDs (1, &(data->platform), &num_platforms);

      if (err_num != CL_SUCCESS || num_platforms <= 0)
        {
          printf ("Failed to find any OpenCL platforms.\n");
          return;
        }

      /* Set up context for GPU */
      err_num = ocl_set_up_context (CL_DEVICE_TYPE_CPU, &(data->platform), &(data->context), &(data->device), &(data->command_queue));
      check_error (err_num, CL_SUCCESS);

      /* Load an build OpenCL program */
      /* FIXME hardcoded file name */
      data->program = load_and_build_program (data->context, data->device,
          "/home/iaguis/igalia/test-skeltrack/ccl.cl");

      /* Device buffers creation */
      data->buffer_matrix_device = clCreateBuffer (data->context,
          CL_MEM_READ_ONLY, sizeof(gint) * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->labels_matrix_device = clCreateBuffer (data->context,
          CL_MEM_READ_WRITE, sizeof(gint) * matrix_size, NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      data->mD_device = clCreateBuffer (data->context, CL_MEM_READ_WRITE,
          sizeof(gint), NULL, &err_num);
      check_error (err_num, CL_SUCCESS);

      /* Create kernels */
      data->initialize_labels = clCreateKernel (data->program,
          "initialize_labels", &err_num);
      check_error (err_num, CL_SUCCESS);

      data->mesh_kernel = clCreateKernel (data->program, "mesh_kernel", &err_num);
      check_error (err_num, CL_SUCCESS);
    }
}

gint round_worksize_up(gint group_size, gint global_size)
{
  gint remainder = global_size % group_size;

  if (remainder == 0)
    {
      return global_size;
    }
  else
    {
      return global_size + group_size - remainder;
    }
}

void
ocl_ccl (oclCclData *data,
         gint *buffer,
         gint width,
         gint height)
{
  gint size;
  cl_int err_num;
  size_t local_worksize[2], global_worksize[2];
  size_t init_worksize;
  cl_event read_done;

  size = width * height;

  data->buffer_matrix = buffer;
  data->labels_matrix = g_slice_alloc (size * sizeof(gint));

  local_worksize[0] = 16;
  local_worksize[1] = 16;
  global_worksize[0] = round_worksize_up(local_worksize[0], width);
  global_worksize[1] = round_worksize_up(local_worksize[1], height);

  err_num = CL_SUCCESS;
  err_num |= clSetKernelArg (data->mesh_kernel, 0, sizeof(cl_mem),
      &(data->buffer_matrix_device));
  err_num |= clSetKernelArg (data->mesh_kernel, 1, sizeof(cl_mem),
      &(data->labels_matrix_device));
  err_num |= clSetKernelArg (data->mesh_kernel, 2, sizeof(cl_mem),
      &(data->mD_device));
  err_num |= clSetKernelArg (data->mesh_kernel, 3, sizeof(gint),
      &width);
  err_num |= clSetKernelArg (data->mesh_kernel, 4, sizeof(gint),
      &height);
  err_num |= clSetKernelArg (data->mesh_kernel, 5, sizeof(gint) *
      local_worksize[0] * local_worksize[1], NULL);
  check_error (err_num, CL_SUCCESS);

  err_num |= clSetKernelArg (data->initialize_labels, 0, sizeof(cl_mem),
      &(data->labels_matrix_device));
  err_num |= clSetKernelArg (data->initialize_labels, 1, sizeof(gint), &size);
  check_error (err_num, CL_SUCCESS);

  // Copy new data to device
  err_num = clEnqueueWriteBuffer (data->command_queue,
      data->buffer_matrix_device, CL_FALSE, 0, sizeof (gint) * size,
      data->buffer_matrix, 0, NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueWriteBuffer (data->command_queue,
      data->labels_matrix_device, CL_FALSE, 0, sizeof (gint) * size,
      data->labels_matrix, 0, NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  err_num = clEnqueueWriteBuffer (data->command_queue, data->mD_device,
      CL_FALSE, 0, sizeof(gint), data->mD, 0, NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  init_worksize = size;

  err_num = clEnqueueNDRangeKernel (data->command_queue,
      data->initialize_labels, 1, NULL, &init_worksize, NULL, 0, NULL, NULL);
  check_error (err_num, CL_SUCCESS);

  while (data->mD)
    {
      err_num = clEnqueueNDRangeKernel (data->command_queue, data->mesh_kernel,
          2, NULL, global_worksize, local_worksize, 0, NULL, NULL);
      check_error (err_num, CL_SUCCESS);


      err_num = clEnqueueReadBuffer (data->command_queue, data->mD_device, CL_FALSE, 0,
      sizeof (gint), data->mD, 0, NULL, &read_done);
      check_error (err_num, CL_SUCCESS);

      clWaitForEvents (1, &read_done);
    }
  err_num = clEnqueueReadBuffer (data->command_queue,
      data->labels_matrix_device, CL_FALSE, 0, sizeof (gint) * size,
      data->labels_matrix, 0, NULL, &read_done);
  check_error (err_num, CL_SUCCESS);

  clWaitForEvents (1, &read_done);

  return;
}

int main(int argc, char **argv)
{
  if (argc < 2) {
    printf("Invalid arguments\n");
    exit(1);
  }

  oclCclData *data = NULL;

  if (clutter_init (&argc, &argv) != CLUTTER_INIT_SUCCESS)
    return -1;

  guint16 *buffer;
  guint width, height, reduced_width, reduced_height;

  width = 640;
  height = 480;

  data = g_slice_alloc0 (sizeof(oclCclData));
  data->mD = g_slice_alloc0 (sizeof(gint));

  buffer = reduce_depth_file(argv[1], 16, &reduced_width, &reduced_height);

  ocl_init (data, reduced_width * reduced_height);

  ocl_ccl (data, buffer, reduced_width, reduced_height);

  int i, j;
  for (i=0; i<reduced_width; i++) {
    for (j=0; j<reduced_height; j++) {
      printf("%d ", data->labels_matrix[j*reduced_width + i]);
    }
    printf("\n");
  }

  g_slice_free (oclCclData, data);

  return 0;
}
