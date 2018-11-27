#include "program.h"


/* *********************************************************************************************************
*	K-Means Algorithm Actual Execution
* ********************************************************************************************************* */
BOOLEAN parallelKMeansExecution(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray, float* resultTime, float* resultQuality)
{
	float time = 0.0;
	int tag;

	/* Loop is synchronized accross processes. */
	for(int i = 0; i < parameters->iterationLimit; i++)
	{
		/* ============================
		 *	SYNC: MASTER -> SLAVE 
		 *	(Point + Clusters)
		 * ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, TRUE, DIRECTION_MASTER_SLAVE, TAG_OK) == FALSE)
		{
			return FALSE;
		}

		/* ========================================================
		 *	STEP 1: 
		 *		Distribute points to clusters
		 * ======================================================== */
		if (parallelDistributePointsToClusters(myId, worldSize, parameters, pointArray, clusterArray) == FALSE)
			tag = TAG_STOP;
		else
			tag = TAG_OK;

		/* ============================
		*	SYNC: SLAVE -> MASTER
		*	(Points)
		*  ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_SLAVE_MASTER, tag) == FALSE)
			tag = TAG_STOP;
		else
			tag = TAG_OK;

		/* ============================
		*	SYNC: MASTER -> SLAVE
		*	(Points)
		*  ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_MASTER_SLAVE, tag) == FALSE)
		{
			PRINT_TERMINATION_CAUSE(FAIL_UNSHIFTED_POINTS)
			return FALSE;
		}
		
		/* ========================================================
		 *	STEP 2:
		 *		Recalculate claster center and diameter
		 * ======================================================== */
		if (parallelRecalculateClusters(myId, worldSize, parameters, pointArray, clusterArray) == FALSE)
			tag = TAG_STOP;
		else
			tag = TAG_OK;

		/* ============================
		*	SYNC: SLAVE -> MASTER
		*	(Clusters)
		* ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, FALSE, TRUE, DIRECTION_SLAVE_MASTER, tag) == FALSE)
			tag = TAG_STOP;
		else
			tag = TAG_OK;

		/* ========================================================
		 *	STEP 3:
		 *		Master calculates quality
		 * ======================================================== */
		if (myId == MASTER_PROCESS && tag == TAG_OK)
		{
			float quality = sequentialQualityCalculation(parameters, clusterArray);
			printf("Iteration: %d - Time: %f - Quality: %f\n", i, time, quality);
			if(quality <= parameters->qualityMeasure)
			{
				*resultQuality = quality;
				*resultTime = time;
				syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, FALSE, TRUE, DIRECTION_MASTER_SLAVE, TAG_STOP);
				PRINT_TERMINATION_CAUSE(SUCCESS)
				return TRUE;
			}
		}

		/* ============================
		*	SYNC: MASTER -> SLAVE
		*	(Clusters)
		* ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, FALSE, TRUE, DIRECTION_MASTER_SLAVE, tag) == FALSE)
		{
			PRINT_TERMINATION_CAUSE(FAIL_CUDA_FAILED)
			return FALSE;
		}

		/* ========================================================
		 *	STEP 4:
		 *		Shift point
		 * ======================================================== */
		parallelRecalculatePointLocations(myId, worldSize, parameters, pointArray);
		time += parameters->timeInterval;

		/* ============================
		*	SYNC: SLAVE -> MASTER
		*	(Points)
		* ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_SLAVE_MASTER, tag) == FALSE)
			tag = TAG_STOP;
		else
			tag = TAG_OK;

		/* ========================================================
		 *	STEP 5:
		 *		Check if time limit reached
		 * ======================================================== */
		if(myId == MASTER_PROCESS && time == parameters->timeLimit)
		{
			/* ============================
			*	SYNC: MASTER -> SLAVE
			*	(Points)
			* ============================ */
			syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_MASTER_SLAVE, TAG_STOP);
			{
				PRINT_TERMINATION_CAUSE(FAIL_TIME_LIMIT_REACHED)
				return FALSE;
			}
		}

		/* ============================
		*	SYNC: MASTER -> SLAVE
		*	(Points)
		* ============================ */
		if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_MASTER_SLAVE, tag) == FALSE)
		{
			return FALSE;
		}
#ifdef __DEBUG_MODE
		DEBUG_ITERATION_PRINT
#endif

	}

	/* ============================
	*	SYNC: MASTER -> SLAVE
	*	(Points)
	* ============================ */
	if (syncProcesses(myId, worldSize, pointArray, clusterArray, parameters, TRUE, FALSE, DIRECTION_MASTER_SLAVE, TAG_STOP) == FALSE)
	{
		PRINT_TERMINATION_CAUSE(FAIL_ITERATION_LIMIT_REACHED)
		return FALSE;
	}
}

/* *********************************************************************************************************
 * Each process distribute its portion of the point array to suitable fitting clusters.
 * return:
 *	TRUE - Atleast 1 point shifted between clusters.
 *	FALSE - No points shifted between clusters.
 * ********************************************************************************************************* */
BOOLEAN parallelDistributePointsToClusters(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray)
{
	int startPointIndex, endPointIndex, i, tid;
	BOOLEAN shifted = FALSE;

	/* Calc the start and end indices of point array. */
	calcPointStartAndEndIndices(myId, worldSize, parameters->numOfPoints, &startPointIndex, &endPointIndex);



	/* Parallel distribution (OMP). */
		#pragma omp parallel for
		for (i = startPointIndex; i < endPointIndex; i++)
		{
				
			int minClusterId = -1;
			float minDistance = -1, tempDistance = -1;

			/* Calculate distance from point i to every cluster, and save min distance. */
			for (int j = 0; j < parameters->numOfClusters; j++)
			{
				tempDistance = sqrt(pow((pointArray[i].x - clusterArray[j].x), 2) + pow((pointArray[i].y - clusterArray[j].y), 2) + pow((pointArray[i].z - clusterArray[j].z), 2));
				if (minDistance < 0 || tempDistance < minDistance)
				{
					minClusterId = clusterArray[j].id;
					minDistance = tempDistance;
				}
			}

			/* Check if new cluster found for the point. */
			if (pointArray[i].clusterId != minClusterId)
			{
				if(shifted != TRUE)
				{
					#pragma omp critical
					shifted = TRUE;
				}

				pointArray[i].clusterId = minClusterId;
			}
		}
	

	return shifted;
}

/* *********************************************************************************************************
*	Calculate cluster centers and diameters.
*	Return: 
*		TRUE - Successful calc. 
*		FALSE - Failed calc.
* ********************************************************************************************************* */
BOOLEAN parallelRecalculateClusters(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray)
{
	int startClusterIndex, endClusterIndex, i;
	BOOLEAN success = TRUE;

	/* Calc Start and end indices for cluster array of current process */
	calcClusterStartAndEndIndices(myId, worldSize, parameters->numOfClusters, &startClusterIndex, &endClusterIndex);

	#pragma omp parallel num_threads(endClusterIndex - startClusterIndex + 1)
	{
		int tid = omp_get_thread_num() + startClusterIndex;
		if (calcClusterCenterAndDiameterCuda(pointArray, &clusterArray[tid], parameters->numOfPoints) != cudaSuccess)
		{
			if (success == TRUE)
			{
				#pragma omp critical
				success = FALSE;
			}
		}
	}

	return success;
}

/* *********************************************************************************************************
*	Calculate quality.
*	return calculated quality
* ********************************************************************************************************* */
float sequentialQualityCalculation(K_MEANS_PARAMS* parameters, CLUSTER* clusterArray)
{
	int i;
	float quality = 0.0;
	int divider = (parameters->numOfClusters * (parameters->numOfClusters - 1));

	/* Parallel quality computation. */
	#pragma omp parallel for num_threads(parameters->numOfClusters)
	for (i = 0; i < parameters->numOfClusters; i++)
	{
		for (int j = 0; j < parameters->numOfClusters; j++)
		{
			if (j == i)
				continue;

			float tempDistance = sqrt(pow((clusterArray[j].x - clusterArray[i].x), 2) + pow((clusterArray[j].y - clusterArray[i].y), 2) + pow((clusterArray[j].z - clusterArray[i].z), 2));
			#pragma omp critical
			quality += (clusterArray[i].diameter / (tempDistance * divider));
		}
	}
	

	return quality;
}

/* *********************************************************************************************************
*	Recalculate point location acording time and velocity.
* ********************************************************************************************************* */
BOOLEAN parallelRecalculatePointLocations(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray)
{
	int startPointIndex, endPointIndex, i;

	/* Calc the start and end indices for */
	calcPointStartAndEndIndices(myId, worldSize, parameters->numOfPoints, &startPointIndex, &endPointIndex);

	#pragma omp parallel for
	for (i = startPointIndex; i < endPointIndex; i++)
	{
		pointArray[i].x = pointArray[i].x + (parameters->timeInterval * pointArray[i].vX);
		pointArray[i].y = pointArray[i].y + (parameters->timeInterval * pointArray[i].vY);
		pointArray[i].z = pointArray[i].z + (parameters->timeInterval * pointArray[i].vZ);
	}

	return TRUE;
}