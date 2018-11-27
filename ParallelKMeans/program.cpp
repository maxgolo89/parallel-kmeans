// Author: Max Golotsvan
// ID: 314148123

#include "program.h"

#pragma warning(disable : 4996)
/* *********************************************************************************************************
*	Main.
* ********************************************************************************************************* */
int main(int argc, char *argv[])
{
	int myid;																// Current process id
	int size;																// World size
	MPI_Status status;														// Recv status holder
	K_MEANS_PARAMS parameters;
	POINT* pointArray;
	CLUSTER* clusterArray;
	int i;
	float resultTime = -1, resultQuality = -1;
	int threadsProvided;
	double startTime, endTime;
	

	/* ======================================================
	*	MPI initiations.
	* ====================================================== */
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threadsProvided);
	if (threadsProvided < MPI_THREAD_MULTIPLE)
	{
		printf("Error: the MPI library doesn't provide the required thread level\n");
		MPI_Abort(MPI_COMM_WORLD, 0);
	}
	startTime = MPI_Wtime();
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	omp_set_num_threads(NUM_OF_THREADS_OMP);

	/* ======================================================
	*	Process 0: 
	*	Reads all information from input file.
	*	Allocates memory for cluster array, as indicated in input file.
	*	Initialize random clusters.
	*	Send the parameters, points and clusters to other processes.
	*  ====================================================== */
	if(myid == MASTER_PROCESS)
	{
		/* Read parameters and points from file */
		if (readFile(&parameters, &pointArray, FILE_PATH_INPUT) == FALSE)
			MPI_Abort(MPI_COMM_WORLD, 1);

		if(parameters.numOfClusters < size)
		{
			PRINT_TOO_MANY_PROC(size, parameters.numOfClusters)
			MPI_Abort(MPI_COMM_WORLD, 1);
		}

		/* Allocate memory for cluster array */
		if(allocClusters(&clusterArray, parameters.numOfClusters) == FALSE)
			MPI_Abort(MPI_COMM_WORLD, 1);

		/* Initialize clusters with random centers */
		if (initializeRandomCluster(clusterArray, parameters.numOfClusters) == FALSE)
			MPI_Abort(MPI_COMM_WORLD, 1);
		
		/* Send all info to all other processes. */
		for(i = 1; i < size; i++)
		{
			MPI_Send(&parameters, sizeof(K_MEANS_PARAMS), MPI_CHAR, i, 0, MPI_COMM_WORLD);
			MPI_Send(pointArray, parameters.numOfPoints * sizeof(POINT), MPI_CHAR, i, 0, MPI_COMM_WORLD);
			MPI_Send(clusterArray, parameters.numOfClusters * sizeof(CLUSTER), MPI_CHAR, i, 0, MPI_COMM_WORLD);
		}
	}
	/* ======================================================
	*	Process X - 0:
	*	Recieve parameters, points and clusters from process 0.
	*  ====================================================== */
	else
	{
		/* Recieve all info from master process */
		MPI_Recv(&parameters, sizeof(K_MEANS_PARAMS), MPI_CHAR, MASTER_PROCESS, 0, MPI_COMM_WORLD, &status);
		
		/* Allocate memory for points and clusters */
		allocPoints(&pointArray, parameters.numOfPoints);
		allocClusters(&clusterArray, parameters.numOfClusters);

		/* Recieve point and cluster arrays from master process */
		MPI_Recv(pointArray, parameters.numOfPoints * sizeof(POINT), MPI_CHAR, MASTER_PROCESS, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(clusterArray, parameters.numOfClusters * sizeof(CLUSTER), MPI_CHAR, MASTER_PROCESS, 0, MPI_COMM_WORLD, &status);
	}

	/* ======================================================
	*	Each process executes parallel K-Mean algorithm
	*  ====================================================== */
	parallelKMeansExecution(myid, size, &parameters, pointArray, clusterArray, &resultTime, &resultQuality);

	/* ======================================================
	*	Process 0:
	*	Write results to file
	*  ====================================================== */
	if(myid == MASTER_PROCESS)
		writeFile(&parameters, clusterArray, &resultTime, &resultQuality, FILE_PATH_OUTPUT);

	/* Each process finish print. */
	printf("Process #%d finished\n", myid);

	/* Free allocated memory. */
	free(pointArray);
	free(clusterArray);
	
	/* Print run time */
	endTime = MPI_Wtime();
	if(myid == MASTER_PROCESS) 	
		printf("Program finished after: %f seconds\n", endTime - startTime);

	MPI_Finalize();
	return 0;
}