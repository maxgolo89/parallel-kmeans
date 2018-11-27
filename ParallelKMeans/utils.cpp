#include "program.h"

/* *********************************************************************************************************
*	Read algorithm parameters and points from file.
* ********************************************************************************************************* */
BOOLEAN readFile(K_MEANS_PARAMS* parameters, POINT** pointsArray, const char* fileName)
{
	int i = 0;

	/* Reset params struct. */
	memset(parameters, 0, sizeof(K_MEANS_PARAMS));

	FILE* inputFile = fopen(fileName, FILE_READ_MODE);

	if (inputFile == NULL)
	{
		printf("Unable to open the file: %s\n", fileName);
		return FALSE;
	}

	/* Read first line of parameters */
	fscanf(inputFile, PATTERN_PARAMS, &parameters->numOfPoints, &parameters->numOfClusters, &parameters->timeLimit, &parameters->timeInterval, &parameters->iterationLimit, &parameters->qualityMeasure);

	/* Validate parameters */
	if (parameters->numOfPoints < 0 || parameters->numOfClusters < 0 || parameters->timeLimit < 0 || parameters->timeInterval < 0 || parameters->iterationLimit < 0 || parameters->qualityMeasure < 0)
	{
		printf("Failed to read parameters from file\n");
		fclose(inputFile);
		return FALSE;
	}

	/* Allocate memory for points array */
	allocPoints(pointsArray, parameters->numOfPoints);

	/* Read point until EOF */
	while (fscanf(inputFile, PATTERN_POINTS, &(*pointsArray)[i].x, &(*pointsArray)[i].y, &(*pointsArray)[i].z, &(*pointsArray)[i].vX, &(*pointsArray)[i].vY, &(*pointsArray)[i].vZ) > 0)
	{
		(*pointsArray)[i].id = i;
		i++;
	}

	fclose(inputFile);

	return TRUE;
}

/* *********************************************************************************************************
*	Write results to file.
* ********************************************************************************************************* */
BOOLEAN writeFile(const K_MEANS_PARAMS* parameters, const CLUSTER* clusters, const float* resultTime, const float* resultQuality, const char* fileName)
{
	int i = 0;

	FILE* outputFile = fopen(fileName, FILE_WRITE_MODE);

	/* Validate file opened */
	if (outputFile == NULL)
	{
		printf("Unable to open the file: %s\n", fileName);
		return FALSE;
	}

	if (resultTime >= 0 && resultQuality > 0)
	{
		/* Write time and quality result */
		fprintf(outputFile, PATTERN_RESULT, *resultTime, *resultQuality);
		fprintf(outputFile, PATTERN_CENTERS);

		/* Write cluster centers */
		for (i = 0; i < parameters->numOfClusters; i++)
			fprintf(outputFile, PATTERN_RESULT_POINTS, clusters[i].x, clusters[i].y, clusters[i].z);
	}
	else
	{
		fprintf(outputFile, PATTERN_ALGO_FAILED);
	}


	fclose(outputFile);

	return TRUE;
}

/* *********************************************************************************************************
*	Allocate memory for cluster array
* ********************************************************************************************************* */
BOOLEAN allocClusters(CLUSTER** clusters, int numOfClusters)
{
	/* Allocate memory for cluster array */
	*clusters = (CLUSTER*)malloc(sizeof(CLUSTER) * numOfClusters);

	if (*clusters != NULL)
		return TRUE;

	return FALSE;
}

/* *********************************************************************************************************
*	Allocate memory for point array
* ********************************************************************************************************* */
BOOLEAN allocPoints(POINT** points, int numOfPoints)
{
	/* Allocate memory for points array */
	*points = (POINT*)malloc(sizeof(POINT) * numOfPoints);

	if (*points != NULL)
		return TRUE;

	return FALSE;
}

/* *********************************************************************************************************
*	Initialize cluster array with random center points
* ********************************************************************************************************* */
BOOLEAN initializeRandomCluster(CLUSTER* clusters, int numOfClusters)
{
	int i = 0;
	time_t t;

	srand((unsigned)time(&t));

	if (clusters == NULL)
		return FALSE;

	/* Initiate random center points for all cluster */
	for (; i < numOfClusters; i++)
	{
		clusters[i].id = i;
		clusters[i].x = rand() % 100;
		clusters[i].x *= rand() % 10 > 5 ? 1 : -1;
		clusters[i].y = rand() % 100;
		clusters[i].y *= rand() % 10 > 5 ? 1 : -1;
		clusters[i].z = rand() % 100;
		clusters[i].z *= rand() % 10 > 5 ? 1 : -1;
		clusters[i].diameter = 0;
	}

	return TRUE;
}

/* *********************************************************************************************************
*	Sync point and cluster arrays accross processes.
*	Direction: 
*		0 - Master -> Slaves
*		1 - Slaves -> Master
*	Return:
*		TRUE - stopSignal == FALSE
*		FALSE - stopSignal == TRUE
* ********************************************************************************************************* */
BOOLEAN syncProcesses(int myid, int worldSize, POINT* pointArray, CLUSTER* clusters, K_MEANS_PARAMS* parameters, BOOLEAN syncPoints, BOOLEAN syncClusters, int direction, int tag)
{
	int startPointIndex, endPointIndex, startClusterIndex, endClusterIndex;
	int stopSignal = 1;
	
	if (worldSize == 1)
		return TRUE;

	/* Sync process by direction argument. */
	switch(direction)
	{
	/* =================================
	*	MASTER --> SLAVE
	*  ================================= */
	case DIRECTION_MASTER_SLAVE:
		/*  =================================
		 *  Sync points: master -> slaves 
		 *  ================================= */
		MPI_Status status;

		if (syncPoints == TRUE)
		{
			if(myid == MASTER_PROCESS)
			{
				/* Each spawn thread responsible for sending to a matching slave process. */
				#pragma omp parallel num_threads(worldSize -1) firstprivate(pointArray)
				{
					int tid = omp_get_thread_num() + 1;
					MPI_Send(pointArray, parameters->numOfPoints * sizeof(POINT), MPI_CHAR, tid, tag, MPI_COMM_WORLD);

				}

				if (tag == TAG_OK)
					stopSignal = FALSE;
			}
			else
			{
				MPI_Recv(pointArray, parameters->numOfPoints * sizeof(POINT), MPI_CHAR, MASTER_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

				if (status.MPI_TAG == TAG_OK)
					stopSignal = FALSE;
			}
		}

		/*  =================================
		 *  Sync clusters: master -> slaves
		 *  ================================= */
		if (syncClusters == TRUE)
		{
			if (myid == MASTER_PROCESS)
			{
				/* Each spawn thread responsible for sending to a matching slave process. */
				#pragma omp parallel num_threads(worldSize -1) firstprivate(clusters)
				{
					int tid = omp_get_thread_num() + 1;
					MPI_Send(clusters, parameters->numOfClusters * sizeof(CLUSTER), MPI_CHAR, tid, tag, MPI_COMM_WORLD);
				}

				if (tag == TAG_OK)
					stopSignal = FALSE;
			}
			else
			{
				MPI_Recv(clusters, parameters->numOfClusters * sizeof(CLUSTER), MPI_CHAR, MASTER_PROCESS, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
				
				if (status.MPI_TAG == TAG_OK)
					stopSignal = FALSE;
			}
		}
		break;
	/* =================================
	*	SLAVE --> MASTER
	*  ================================= */
	case DIRECTION_SLAVE_MASTER:
		/*  =================================
		 *  Sync points: slaves -> master
		 *  ================================= */
		if (syncPoints == TRUE)
		{
			if(myid == MASTER_PROCESS)
			{
				/* Each spawn thread responsible for recieving from a matching slave process. */
				#pragma omp parallel num_threads(worldSize -1)
				{
					int tid = omp_get_thread_num() + 1;
					int startPointIndexRemoteProcess, endPointIndexRemoteProcess;
					MPI_Status status;

					/* Calc remote indices to recieve, and recieve. */
					calcPointStartAndEndIndices(tid, worldSize, parameters->numOfPoints, &startPointIndexRemoteProcess, &endPointIndexRemoteProcess);
					MPI_Recv(&pointArray[startPointIndexRemoteProcess], (endPointIndexRemoteProcess - startPointIndexRemoteProcess + 1) * sizeof(POINT), MPI_CHAR, tid, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
					
					if(status.MPI_TAG == TAG_OK)
					{
						if(stopSignal == TRUE)
						{
							#pragma omp critical
							stopSignal = FALSE;
						}
					}
				}

//				MPI_Status status;
//				for(int i = 1; i < worldSize; i++)
//				{
//					int startPointIndexRemoteProcess, endPointIndexRemoteProcess;
//					calcPointStartAndEndIndices(i, worldSize, parameters->numOfPoints, &startPointIndexRemoteProcess, &endPointIndexRemoteProcess);
//					MPI_Recv(&pointArray[startPointIndexRemoteProcess], (endPointIndexRemoteProcess - startPointIndexRemoteProcess + 1) * sizeof(POINT), MPI_CHAR, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
//
//					if (status.MPI_TAG == TAG_OK)
//					{
//						if (stopSignal == TRUE)
//						{
//							stopSignal = FALSE;
//						}
//					}
//				}
			}
			else
			{
				/* Calc indices to send, and send. */
				calcPointStartAndEndIndices(myid, worldSize, parameters->numOfPoints, &startPointIndex, &endPointIndex);
				MPI_Send(&pointArray[startPointIndex], (endPointIndex - startPointIndex + 1) * sizeof(POINT), MPI_CHAR, MASTER_PROCESS, tag, MPI_COMM_WORLD);
				if (tag == TAG_OK)
					stopSignal = FALSE;
			}
		}

		/*  =================================
		 *  Sync clusters: slaves -> master
		 *  ================================= */
		if (syncClusters == TRUE)
		{
			if (myid == MASTER_PROCESS)
			{
				/* Each spawn thread responsible for recieving from a matching slave process. */
				#pragma omp parallel num_threads(worldSize -1)
				{
					int tid = omp_get_thread_num() + 1;
					int startClusterIndexRemoteProcess, endClusterIndexRemoteProcess;
					MPI_Status status;

					/* Calc remote indices to recieve, and recieve. */
					calcClusterStartAndEndIndices(tid, worldSize, parameters->numOfClusters, &startClusterIndexRemoteProcess, &endClusterIndexRemoteProcess);
					MPI_Recv(&clusters[startClusterIndexRemoteProcess], (endClusterIndexRemoteProcess - startClusterIndexRemoteProcess + 1) * sizeof(CLUSTER), MPI_CHAR, tid, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

					if (status.MPI_TAG == TAG_OK)
					{
						if (stopSignal == TRUE)
						{
							#pragma omp critical
							stopSignal = FALSE;
						}
					}
				}
			}
			else
			{
				/* Calc indices to send, and send. */
				calcClusterStartAndEndIndices(myid, worldSize, parameters->numOfClusters, &startClusterIndex, &endClusterIndex);
				MPI_Send(&clusters[startClusterIndex], (endClusterIndex - startClusterIndex + 1) * sizeof(CLUSTER), MPI_CHAR, MASTER_PROCESS, tag, MPI_COMM_WORLD);

				if (tag == TAG_OK)
					stopSignal = FALSE;
			}
		}
		break;
	default:
		break;
	}


	if (stopSignal == TRUE)
		return FALSE;

	return TRUE;
}

/* *********************************************************************************************************
*	Calculate the cluster's array and point's array start and end indices for specific id number
* ********************************************************************************************************* */
BOOLEAN calcPointStartAndEndIndices(int id, int worldSize, int numOfPoints, int* startPointIndex, int* endPointIndex)
{
	/* Calc start and end indices for point array for current process */
	*startPointIndex = (numOfPoints / worldSize) * id;
	*endPointIndex = (numOfPoints / worldSize) * (id + 1) - 1;
	if (id == worldSize - 1 && numOfPoints % worldSize != 0)
		*endPointIndex += numOfPoints % worldSize;

	return TRUE;
}

BOOLEAN calcClusterStartAndEndIndices(int id, int worldSize, int numOfClusters, int* startClusterIndex, int* endClusterIndex)
{
	/* Calc start and end indices for cluster array for current process */
	*startClusterIndex = (numOfClusters / worldSize) * id;
	*endClusterIndex = (numOfClusters / worldSize) * (id + 1) - 1;
	if (id == worldSize - 1 && numOfClusters % worldSize != 0)
		*endClusterIndex += numOfClusters % worldSize;

	return TRUE;
}