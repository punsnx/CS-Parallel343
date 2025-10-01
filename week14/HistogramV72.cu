
#define NUM_BINS 128


__global__ void histogram_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elemets, unsigned int numbins)
{
	// Privatized bins
	// To implement

	// Histogram
	// To implement

	// Commit global memory
	// To implement
}

void histogram(unsigned int* input, unsigned int* bins, 
		unsigned int num_elements, unsigned int num_bins)
{
	// Set each bin counter to zero
	// To implement

	histogram_kernel<<< 30, 512 >>>(input, bins, num_elements, num_bins);
}	

int main()
{
	int inputLength;
	unsigned int* hostInput;
	unsigned int* hostBins;
	
	unsigned int* deviceInput;
	unsigned int* deviceBins;	

	// Init hostInput & hostBins


	// Allocate device memory

	
	// Copy host memory to device


	// initiailize thread block and grid dimensions
	// invoke CUDA kernel
	histogram(deviceInput, deviceBins, inputLength, NUM_BINS);


	// Copy results from device to host

	// deallocate device memory


}
