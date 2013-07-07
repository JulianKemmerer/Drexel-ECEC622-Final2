/*  Device code for the Trap, yo */
#define THREADS_PER_BLOCK 128
#define NUM_TRAPEZOIDS 8000000

#ifndef _TRAP_KERNEL_H_
#define _TRAP_KERNEL_H_

//ITS A TRAP!
/*------------------------------------------------------------------
 * Function:    f - for function
 * Purpose:     Compute a value that will knock your socks off
 * Input args:  x - yep, that's it.
 * Output: (x+1)/sqrt(x*x + x + 1) or whateva, eh.
 */
//Ready for this?
//Function as a macro...
//What what!?
//Avoids overhead of function calls
#define f(x) (((x) + 1)/sqrt((x)*(x) + (x) + 1))
//Though somewhere I did read that cuda inlines all functions anyway...

__device__ void kahan_sum(float * result, float  * input, int start_index, int end_index, int increment)
{
	/* Thanks wikipedia
	    function KahanSum(input)
		var sum = 0.0
		var c = 0.0          //A running compensation for lost low-order bits.
		for i = 1 to input.length do
			var y = input[i] - c    //So far, so good: c is zero.
			var t = sum + y         //Alas, sum is big, y small, so low-order digits of y are lost.
			c = (t - sum) - y   //(t - sum) recovers the high-order part of y; subtracting y recovers -(low part of y)
			sum = t             //Algebraically, c should always be zero. Beware overly-aggressive optimising compilers!
			//Next time around, the lost low part will be added to y in a fresh attempt.
		return sum
	*/
	
	//Volatile to avoid being compiled out
	volatile float sum = 0.0;
	volatile float c = 0.0;
	int i;
	for(i=start_index; i <=end_index; i+=increment)
	{
		float y = input[i] - c;
		float t = sum + y;
		c = (t-sum) - y;
		sum = t;
	}
	*result = sum;
	
	
	//Original non-kahan
	/*
	float sum = 0.0;
	int i;
	for(i=start_index; i <=end_index; i+=increment)
	{
		sum += input[i];
	}
	*result = sum;
	*/
}


__global__ void trap_kernel_slow(float a, float b, int n, float h, float * results)
{
	//Each thread will do one k iteration
	//Get a thread id
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	//k starts at one so add one
	int k = tx + 1;
	
	float x = a+k*h;
	results[tx] = f(x);
	
	//That's all folks
	//Summing in done in seperate kernel
}

__global__ void trap_kernel_slow_sum(float * sum,float a, float b, int n, float h, float * results)
{
	//Sum over results in global mem
	//Only one thread should do this
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	if(tx==0)
	{
		//Loop over results and sum them
		float integral;
		int k;
		integral = (f(a) + f(b))/2.0;
		for (k = 1; k <= n-1; k++) {
			integral += results[k-1];
		}
		integral = integral*h;
		*sum = integral;
	}
}

__global__ void trap_kernel_fast_sum(float a, float b, float h, float * end_result, int global_mem_floats_per_thread, int global_mem_floats, float * global_mem_float_array)
{
	//Thread id
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	
	//First part is to sum/collect global memory values into shared mem
	//There is only one block
	//global_mem_floats_per_thread is the number of floats
	//Calculate this threads range
	// tx 0, 0 to (global_mem_floats_per_thread-1)
	// tx 1, global_mem_floats_per_thread to ...
	int global_mem_start_index = (tx *global_mem_floats_per_thread);
	int global_mem_end_index = global_mem_start_index + global_mem_floats_per_thread -1;
	//There are maximum limits (more threads are created then needed...)
	//Only go up to the number of global mem floats (-1 for index)
	int global_mem_index_max = global_mem_floats-1;
	
	//Summming over values in global_mem_float_array
	float thread_result = 0;
	//A weeee bit of thread divergence
	//Keeps us from summing too far
	int summing_index_max;
	if(global_mem_end_index<=global_mem_index_max)
	{
		//No range limiting
		summing_index_max = global_mem_end_index;
	}
	else
	{
		//Limit range to max
		summing_index_max = global_mem_index_max;
	}
	//Each thread does Kahan sum for these global memory vals
	kahan_sum(&thread_result, global_mem_float_array,global_mem_start_index, summing_index_max,1);
	
	//This threads result is in thread_result
	//Copy this into spot in shared mem
	//Initially shared mem must be the size the number of threads used
	//As each thread produces a result 
	//(and only one block so = threads per block)
	__shared__ float thread_vals[THREADS_PER_BLOCK];
	thread_vals[tx] = thread_result;
	
	//Sync here before next step (one block so all threads sync)
	__syncthreads();
	
	//I implemented a tree based sum
	//Even made this nice figure...
	/*
		| ----x  ---  |	
		 *      *       *       *       *       *       *       *
		|s	s|	s	s |	s	s|	s	s|	s	s|	s	s|	s	s|	s 	s|
		|s      s     | s       s    |  s       s    |  s       s    |  <<Level 1
		|s				s			 |  s               s            |	<<Level 2
		|s							    s							 |	<<Level 3
		 s																<<Level 4
	*/
	//But!... there are so little things to sum at this step
	// that it was a big waste of time
	//As doing the sum in just one thread is just as fast
	//So here it is with just one thread:
	//I can provide the other code if requested
	if(tx==0)
	{
		float ret_val = 0;
		kahan_sum(&ret_val, &(thread_vals[0]),0,THREADS_PER_BLOCK-1, 1);
		*end_result = (ret_val + ((f(a) + f(b))/2.0))*h;
		return;
	}
	else
	{
		return;
	}
}

__global__ void trap_kernel_fast(int n, float a, float h, float * gpu_block_results)
{
	//Each thread does one iteration, placing value in shared mem
	//Get a thread id
	int tx = blockIdx.x * blockDim.x + threadIdx.x;
	//k starts at one so add one
	int k = tx + 1;
	float x = a+k*h;
	float val = f(x);
	
	//Write this into shared mem location
	//Shared mem for this thread block
	//Each thread writes one float so need that many
	__shared__ float thread_vals[THREADS_PER_BLOCK];
	//Use thread index to write into block shared mem
	thread_vals[threadIdx.x] = val;
	
	//Sync all threads in this block
	__syncthreads();
	
	//One thread from this block sums vals into its local mem
	float thread_block_sum = 0.0;
	if(threadIdx.x == 0)
	{
		//Sum values in shared mem, store in local mem
		//Don't sum too far, some threads are uneeded
		//Individual threads, (tx's) span a range
		//Only some of them are valid
		//Originally (k = 1; k <= n-1; k++)
		//Which is tx = 0 to tx <= (n-2)
		//int min_tx = 0;
		int max_tx = (n-2);
		//Calculate equivalent tx range for this block
		//Overall low and high tx (thread id/index) bound for this block
		int low_tx = blockIdx.x * blockDim.x;
		int high_tx = blockIdx.x * blockDim.x + THREADS_PER_BLOCK - 1;
		//Make sure this block is in range at all 
		if(low_tx > max_tx)
		{
			//This block is out of range completely, unneeded
			//Do nothing
			return;
		}
		else
		{
			//Block is partially in range, check top
			if(high_tx > max_tx)
			{
				//High range goes too high
				//Limit to max
				high_tx = max_tx;
			}
			
			//Now have high and low tx values
			//Convert these back to indices into block shared thread_vals[]
			// ( 0->THREADS_PER_BLOCK-1 values )
			int idx_start = low_tx - (blockIdx.x * blockDim.x); //0 always
			int idx_end = high_tx - (blockIdx.x * blockDim.x);
			kahan_sum(&thread_block_sum, &(thread_vals[0]), idx_start, idx_end,1);
		}
		
		//Same thread writes that local mem to a global mem spot
		gpu_block_results[blockIdx.x] = thread_block_sum;
	}
}

#endif // #ifndef _TRAP_KERNEL_H_
