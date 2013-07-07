/*  Purpose: Calculate definite integral using trapezoidal rule.
 *
 * Input:   a, b, n
 * Output:  Estimate of integral from a to b of f(x)
 *          using n trapezoids.
 *
 * Compile: gcc -g -Wall -o trap trap.c -fopenmp
 * Usage:   ./trap
 *
 * Note:    The function f(x) is hardwired.
 *
 */

#ifdef _WIN32
#  define NOMINMAX 
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <trap_kernel.cu>

//Err
void check_fo_error(char *msg);

//Globals for time
float cpu_time;
float gpu_slow;
float gpu_fast;
//Timer for timing time measured in time units
unsigned int timer;

#define LEFT_ENDPOINT 5
#define RIGHT_ENDPOINT 1000
//#define NUM_TRAPEZOIDS 100000 //MOVED TO _kernel.cu file

float compute_on_device(float , float , int, float ,int);
extern "C" float compute_gold(float , float , int, float );

int main(void) 
{
	int n = NUM_TRAPEZOIDS;
	float a = LEFT_ENDPOINT;
	float b = RIGHT_ENDPOINT;
	float h = (b-a)/(float)n; // Height of each trapezoid  
	printf("The height of the trapezoid is %f \n", h);
	printf("Number of trapezoids: %d\n",n);
	
	//Start timer...NOW~!~!~!~!
	cutCreateTimer(&timer);
	cutStartTimer(timer);
	float reference = compute_gold(a, b, n, h);
	printf("Reference solution computed on the CPU = %f \n", reference);
	//Stop timer				 
	cutStopTimer(timer);
	cpu_time = 1e-3 * cutGetTimerValue(timer);
	
	float gpu_result_slow = compute_on_device(a, b, n, h, 1);
	printf("	Solution computed on the GPU (slow) = %f \n", gpu_result_slow);			
	float slow_per_diff = 100*(fabs(gpu_result_slow -reference)/reference);
	printf("	Percent difference computed on the GPU (slow) = %f \n", slow_per_diff );
	
	float gpu_result_fast = compute_on_device(a, b, n, h, 0);
	printf("	Solution computed on the GPU (fast) = %f \n", gpu_result_fast);
	float fast_per_diff = 100*(fabs(gpu_result_fast -reference)/reference);
	printf("	Percent difference computed on the GPU (fast) = %f \n", fast_per_diff );
	
	//Speedup inffffffooo
	printf("== Speedup Info == \n");
	printf("CPU Run time:    %0.10f s. \n", cpu_time);
	printf("GPU (slow) Run time:    %0.10f s. \n", gpu_slow);
	printf("GPU (fast) Run time:    %0.10f s. \n", gpu_fast);
	float gpu_slow_speedup = cpu_time/gpu_slow;
	float gpu_fast_speedup = cpu_time/gpu_fast;
	printf("GPU (slow) Speedup: %0.10f\n", gpu_slow_speedup);
	printf("GPU (fast) Speedup: %0.10f\n",gpu_fast_speedup );
	
	//For copy paste
	//printf("%d	%f	%f	%f	%f\n", NUM_TRAPEZOIDS,gpu_fast_speedup,gpu_slow_speedup,fast_per_diff,slow_per_diff);
}

/* Complete this function to perform the trapezoidal rule on the GPU. */
float compute_on_device(float a, float b, int n, float h, int do_slow)
{		
	//Allocate space on gpu
	//Room for the sum
	float * sum_on_gpu;
	int result_size = 1 * sizeof(float);
	cudaMalloc(&sum_on_gpu, result_size);
	
	//Calculate kernel params
	//How many threads per block?
	int threads_per_block = THREADS_PER_BLOCK;
	//How many calculations?
	//Defined by for (k = 1; k <= n-1; k++)
	//Example n=5, k=1,2,3,4  so num calcs total = n-1
	int total_calcs = n-1;
	//Each thread will take a single calculation
	int calcs_per_thread = 1;
	//How many threads to use?
	int num_threads = ceil((float)total_calcs / (float)calcs_per_thread);
	//How many blocks?
	int num_blocks = ceil((float)num_threads / (float)threads_per_block);
	dim3 thread_block(threads_per_block, 1, 1);
	dim3 grid(num_blocks,1);
	
	
	//Call kernel
	//do_slow indicates if this should execute the slow or fast kernel
	if(do_slow)
	{
		printf("== Slow Kernel==\n");
		
		//Also allocate space for dumb storage of each threads result
		float * gpu_thread_results;
		//All threads (useful or not) will calculate a result to avoid
		//warp divergence...even though this is so slow it won't matter
		cudaMalloc(&gpu_thread_results, num_blocks*threads_per_block * sizeof(float));
		
		//Start timer...NOW~!~!~!~!
		cutCreateTimer(&timer);
		cutStartTimer(timer);
		
		//Slow kernel
		trap_kernel_slow<<<grid, thread_block>>>(a, b, n, h,gpu_thread_results);
		
		//Sync
		cudaThreadSynchronize();
		check_fo_error("trap_kernel_slow FAILURE");
		
		//Summing slow kernel
		//REALLY slow...only one thread does summing
		//Obviously will be made better in optimized version
		dim3 thread_block_slow_sum(1, 1, 1);
		dim3 grid_slow_sum(1,1);
		trap_kernel_slow_sum<<<grid_slow_sum, thread_block_slow_sum>>>(sum_on_gpu, a, b, n, h, gpu_thread_results);
		
		//Sync at end
		cudaThreadSynchronize();
		check_fo_error("trap_kernel_slow_sum FAILURE");
		
		//Stop timer				 
		cutStopTimer(timer);
		gpu_slow = 1e-3 * cutGetTimerValue(timer);
	}
	else
	{
		printf("== Fast Kernel==\n");
		//Each thread does one iteration, placing value in shared mem
		//One thread from each block sums shared mem and writes to global
		
		//Threads in block will each take one iteration
		//One thread form block will sum all threads results
		//Same one thread will write into global memory
		float * gpu_block_results;
		cudaMalloc(&gpu_block_results, num_blocks * sizeof(float));
		
		//Start timer...NOW~!~!~!~!
		cutCreateTimer(&timer);
		cutStartTimer(timer);
		
		//Fast kernel
		trap_kernel_fast<<<grid, thread_block>>>(n,a,h,gpu_block_results);
		
		//Sync
		cudaThreadSynchronize();
		check_fo_error("trap_kernel_fast FAILURE");
		
		//DEBUG//////
		/*float result = 0;
		cudaMemcpy(&result, gpu_block_results, result_size, cudaMemcpyDeviceToHost);
		printf("sum_on_gpu before fast sum: %f\n", result);
		float tmp_result = (result + ((f(a) + f(b))/2.0))*h;
		printf("result before fast sum: %f\n", tmp_result);
		//////////////////////
		*/
		
		//Use another kernel to sum block results in global mem
		//There are now 'num_blocks' floats in global memory
		int global_mem_floats = num_blocks;
		//To sync summation and use only shared memory we can only have
		//one thread block
		num_blocks = 1;
		//threads per block
		threads_per_block = THREADS_PER_BLOCK;
		num_threads = threads_per_block * num_blocks;
		
		//Use tree based approach
		//First level of tree is reading from global mem
		//Want second level to be in shared mem (get out of global as
		// soon as possible)
		/*
			|-----x------|
			|g	g	g	g|	g	g	g	g|	g	g	g	g|	<<< Single block
				   s              s               s			<<< condensing/summing into shared mem
		*/
		//How many floats can fit in shared mem?
		int shared_mem_per_block_size = 16384;
		int max_floats_per_block = shared_mem_per_block_size / sizeof(float);
		
		//Each thread sums some global floats into shared mem
		int global_floats_per_thread = ceil((float)global_mem_floats/(float)num_threads);
		
		dim3 thread_block_fast_sum(threads_per_block, 1, 1);
		dim3 grid_fast_sum(num_blocks,1);

		//Fast sum kernel
		trap_kernel_fast_sum<<<grid_fast_sum, thread_block_fast_sum>>>(a,b,h,sum_on_gpu, global_floats_per_thread, global_mem_floats, gpu_block_results);
				
		//Sync
		cudaThreadSynchronize();
		check_fo_error("trap_kernel_fast_sum FAILURE");		
		
		//Stop timer				 
		cutStopTimer(timer);
		gpu_fast = 1e-3 * cutGetTimerValue(timer);
		
		printf("	max_floats_per_block: %d\n",max_floats_per_block);
		printf("	global_mem_floats: %d\n",global_mem_floats);
		printf("	global_floats_per_thread: %d\n",global_floats_per_thread);
	}
	
	printf("	Threads per block: %d\n",threads_per_block);
	printf("	Number of blocks: %d\n",num_blocks);
	int actual_threads = threads_per_block*num_blocks;
	printf("	Number of threads: %d\n",actual_threads);
	
	//Copy back the result
	float result = 0;
    cudaMemcpy(&result, sum_on_gpu, result_size, cudaMemcpyDeviceToHost);
	
	//Return result
	return result;
}


//Error helper
void check_fo_error(char *msg){
	cudaError_t err = cudaGetLastError();
	if(cudaSuccess != err){
		printf("CUDA ERROR: %s (%s). \n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
} 



