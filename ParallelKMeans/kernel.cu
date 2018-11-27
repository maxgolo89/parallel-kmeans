
#include "program.h"

/* *********************************************************************************************************
*	Cuda function entry point.
*	Calculate a given cluster center point, and diameter.
* ********************************************************************************************************* */
cudaError_t calcClusterCenterAndDiameterCuda(POINT* pointArray, CLUSTER* cluster, int pointArraySize)
{
	POINT* pointArrayCuda;
	CLUSTER* clusterCuda;
	double* threadPointSumXYZ;
	double* threadMaxDiameter;
	int* threadNumOfPointsInCluster;
	int blockCountForDiameterCalculation = 0;
	int threadCountForDiameterCalculation = 0;
	cudaError_t cudaStatus;

	/* Block and Thread count parameters for diameter calculation. */
	if(pointArraySize <= NUM_OF_THREADS_PER_BLOCK_CUDA)
	{
		blockCountForDiameterCalculation = 1;
		threadCountForDiameterCalculation = pointArraySize;
	}
	else
	{
		blockCountForDiameterCalculation = (pointArraySize / NUM_OF_THREADS_PER_BLOCK_CUDA);
		blockCountForDiameterCalculation += (pointArraySize % NUM_OF_THREADS_PER_BLOCK_CUDA != 0 ? 1 : 0);
		threadCountForDiameterCalculation = NUM_OF_THREADS_PER_BLOCK_CUDA;
	}

	/* Choose which GPU to run on, change this on a multi-GPU system. */
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n");
		goto Error;
	}
	
	/* Memory Allocation on Device */
	cudaStatus = allocMem(&pointArrayCuda, &clusterCuda, pointArraySize, &threadPointSumXYZ, &threadMaxDiameter, &threadNumOfPointsInCluster);
	if (cudaStatus != cudaSuccess) {
		printf("allocMem failed!  Memory allocation on device failed.\n");
		goto Error;
	}

	/* Copy Memory From Host to Device */
	cudaStatus = copyMem(pointArray, cluster, pointArraySize, pointArrayCuda, clusterCuda, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("copyMem failed!  Memory copy to device failed.\n");
		goto Error;
	}

	/* ===========================================================================
	 *	CUDA: Parallel point summarization
	 * =========================================================================== */
	cudaSumPointPerThread <<<CUDA_SINGLE_BLOCK, NUM_OF_THREADS_PER_BLOCK_CUDA >>>(pointArrayCuda, clusterCuda, pointArraySize, threadPointSumXYZ, threadNumOfPointsInCluster);

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	 * any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching calcCenter!\n", cudaStatus);
		goto Error;
	}

	/* ===========================================================================
	 *	CUDA: Summarize parallel point computation from each thread
	 * =========================================================================== */
	cudaCalcCenterTotal <<<CUDA_SINGLE_BLOCK, CUDA_SINGLE_THREAD >>>(clusterCuda, threadPointSumXYZ, threadNumOfPointsInCluster);

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	 * any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching calcCenter!\n", cudaStatus);
		goto Error;
	}

	/* ===========================================================================
	 *	CUDA: Parallel diameter calculation
	 * =========================================================================== */
	cudaCalcDiameterPerThread <<<blockCountForDiameterCalculation, threadCountForDiameterCalculation >>> (pointArrayCuda, clusterCuda, pointArraySize, threadMaxDiameter);

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	 * any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching calcDiameter!\n", cudaStatus);
		goto Error;
	}

	/* ===========================================================================
	 *	CUDA: Summarize parallel diameter computation from each thread
	 * =========================================================================== */
	cudaCalcDiameterTotal <<<CUDA_SINGLE_BLOCK, CUDA_SINGLE_THREAD >>> (clusterCuda, threadMaxDiameter, pointArraySize);

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns
	 * any errors encountered during the launch. */
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching calcDiameter!\n", cudaStatus);
		goto Error;
	}

	/* Copy Memory From Device to Host */
	cudaStatus = copyMem(pointArrayCuda, clusterCuda, pointArraySize, pointArray, cluster, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("copyMem failed!  Memory copy to host failed.\n");
		goto Error;
	}

Error:
	cudaFree(pointArrayCuda);
	cudaFree(clusterCuda);

	return cudaStatus;
}

/* *********************************************************************************************************
*	Allocate memory on cuda device
* ********************************************************************************************************* */
cudaError_t allocMem(POINT** pointArrayCuda, CLUSTER** clusterCuda, const int pointArraySize, double** threadPointSumXYZ, double** threadMaxDiameter, int** threadNumOfPointsInCluster)
{
	cudaError_t cudaStatus;

	cudaStatus = cudaMalloc((void**)pointArrayCuda, sizeof(POINT) * pointArraySize);
	if(cudaStatus != cudaSuccess)
		return cudaStatus;

	cudaStatus = cudaMalloc((void**)clusterCuda, sizeof(CLUSTER) * 1);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	cudaStatus = cudaMalloc((void**)threadPointSumXYZ, sizeof(double) * NUM_OF_THREADS_PER_BLOCK_CUDA * 3);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	cudaStatus = cudaMalloc((void**)threadMaxDiameter, sizeof(double) * pointArraySize);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	cudaStatus = cudaMalloc((void**)threadNumOfPointsInCluster, sizeof(int) * NUM_OF_THREADS_PER_BLOCK_CUDA);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;
	
	return cudaSuccess;
}

/* *********************************************************************************************************
*	Copy data from host to cuda device
* ********************************************************************************************************* */
cudaError_t copyMem(POINT* pointArrayFrom, CLUSTER* clusterFrom, int pointArraySizeFrom, POINT* pointArrayTo, CLUSTER* clusterTo, cudaMemcpyKind direction)
{
	cudaError_t cudaStatus;
	
	cudaStatus = cudaMemcpy(pointArrayTo, pointArrayFrom, sizeof(POINT) * pointArraySizeFrom, direction);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	cudaStatus = cudaMemcpy(clusterTo, clusterFrom, sizeof(CLUSTER), direction);
	if (cudaStatus != cudaSuccess)
		return cudaStatus;

	return cudaSuccess;
}

/* *********************************************************************************************************
*	Each thread summarize its portion of points, into a shared array, devided into thread sections.
* ********************************************************************************************************* */
__global__ void cudaSumPointPerThread(POINT* pointArrayCuda, CLUSTER* clusterCuda, int pointArraySizeCuda, double* threadPointSumXYZ, int* threadNumOfPointsInCluster)
{
	int threadId = threadIdx.x;
	int startIndexPerThread, endIndexPerThread, i;

	/* Reset the sum for each thread */
	threadPointSumXYZ[X * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] = 0.0;
	threadPointSumXYZ[Y * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] = 0.0;
	threadPointSumXYZ[Z * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] = 0.0;
	threadNumOfPointsInCluster[threadId] = 0;

	/* Calculate start and end index for each cuda thread
	 * Last thread gets the remainder if not dividable */
	startIndexPerThread = threadId * (pointArraySizeCuda / NUM_OF_THREADS_PER_BLOCK_CUDA);
	endIndexPerThread = startIndexPerThread + (pointArraySizeCuda / NUM_OF_THREADS_PER_BLOCK_CUDA);
	if (threadId == NUM_OF_THREADS_PER_BLOCK_CUDA - 1)
		endIndexPerThread += (pointArraySizeCuda % NUM_OF_THREADS_PER_BLOCK_CUDA);

	/* Sum all points */
	for(i = startIndexPerThread; i < endIndexPerThread; i++)
	{
		if (pointArrayCuda[i].clusterId != clusterCuda->id)
			continue;
			
		threadPointSumXYZ[X * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] += pointArrayCuda[i].x;
		threadPointSumXYZ[Y * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] += pointArrayCuda[i].y;
		threadPointSumXYZ[Z * NUM_OF_THREADS_PER_BLOCK_CUDA + threadId] += pointArrayCuda[i].z;
		threadNumOfPointsInCluster[threadId]++;
	}
}

/* *********************************************************************************************************
*	Single thread summerize the all the summed up points, and calculates the center point
* ********************************************************************************************************* */
__global__ void cudaCalcCenterTotal(CLUSTER* clusterCuda, double* threadPointSumXYZ, int* threadNumOfPointsInCluster)
{
	double sumX = 0.0, sumY = 0.0, sumZ = 0.0;
	int numberOfPointsInCluster = 0, i;

	/* Sum calculations of all threads */
	for (i = 0; i < NUM_OF_THREADS_PER_BLOCK_CUDA; i++)
	{
		sumX += threadPointSumXYZ[X * NUM_OF_THREADS_PER_BLOCK_CUDA + i];
		sumY += threadPointSumXYZ[Y * NUM_OF_THREADS_PER_BLOCK_CUDA + i];
		sumZ += threadPointSumXYZ[Z * NUM_OF_THREADS_PER_BLOCK_CUDA + i];
		numberOfPointsInCluster += threadNumOfPointsInCluster[i];
	}

	if (numberOfPointsInCluster > 0)
	{
		/* Find new center */
		clusterCuda->x = (sumX / numberOfPointsInCluster);
		clusterCuda->y = (sumY / numberOfPointsInCluster);
		clusterCuda->z = (sumZ / numberOfPointsInCluster);
	}
}

/* *********************************************************************************************************
*	Each thread is responsible on a portion of points.
*	Each thread, calculates the max diameter among his points.
* ********************************************************************************************************* */
__global__ void cudaCalcDiameterPerThread(POINT* pointArrayCuda, CLUSTER* clusterCuda, int pointArraySizeCuda, double* threadMaxDiameter)
{
	int threadId = threadIdx.x + (blockIdx.x * blockDim.x);
	double maxDiameterTemp;
	int startIndexPerThread, endIndexPerThread, endComparantIndexThread, j, z;

	/* Reset max diameter placeholders */
	threadMaxDiameter[threadId] = 0.0;
	maxDiameterTemp = 0.0;


	/* Check if threadId doesn't exeed the point array size, if so, it's useless for us. 
	 * Check that the point belongs to the calculated cluster. */
	if (threadId < pointArraySizeCuda && pointArrayCuda[threadId].clusterId != clusterCuda->id)
	{
		endComparantIndexThread = threadId + (pointArraySizeCuda / 2);

		/* Measure max distance from point i to all point from i+1 to i + size/2 (round) */
		for (j = threadId + 1; j < endComparantIndexThread; j++)
		{
			/* True index of comparant, if out of bounds, continue count from beginning of array. */
			z = j % pointArraySizeCuda;

			/* If point doesn't belong to the cluster, skip iteration (Redundent) */
			if (pointArrayCuda[z].clusterId != clusterCuda->id)
				continue;

			/* Distance formula */
			maxDiameterTemp = sqrt(pow((pointArrayCuda[z].x - pointArrayCuda[threadId].x), 2) + pow((pointArrayCuda[z].y - pointArrayCuda[threadId].y), 2) + pow((pointArrayCuda[z].z - pointArrayCuda[threadId].z), 2));

			/* If calculate distance is larger than max diameter, set new max diameter */
			if (maxDiameterTemp > threadMaxDiameter[threadId])
				threadMaxDiameter[threadId] = maxDiameterTemp;
		}
	}
}

/* *********************************************************************************************************
*	Single thread finds the max diameter among calculated max diameters.
* ********************************************************************************************************* */
__global__ void cudaCalcDiameterTotal(CLUSTER* clusterCuda, double* threadMaxDiameter, int threadCount)
{
	int i;

	clusterCuda->diameter = 0.0;
	/* Sum calculations of all threads */
	for (i = 0; i < threadCount; i++)
	{
		if (threadMaxDiameter[i] > clusterCuda->diameter)
			clusterCuda->diameter = threadMaxDiameter[i];
	}
}