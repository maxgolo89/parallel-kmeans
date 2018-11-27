#pragma once
#ifndef _PROGRAM_H
#define _PROGRAM_H

/********************************************************************
*****					PREPROCESSING							*****
********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime_api.h>
#include <ctime>
#include <math.h>




/********************************************************************
*****					ENUM & STRUCTS							*****
********************************************************************/
/* Boolean */
typedef enum
{
	FALSE,
	TRUE
} BOOLEAN;

/* Point */
typedef struct
{
	int id;				// ID assigned to each point (recognizable accross processes).
	int clusterId;		// Cluster ID to which each point belongs to.
	float x;
	float y;
	float z;
	float vX;
	float vY;
	float vZ;
} POINT;

/* Cluster */
typedef struct
{
	int id;				// ID assigned to each cluster (recognizable accross processes).
	float diameter;		// Calculated diameter for each cluster.
	float x;
	float y;
	float z;
} CLUSTER;

/* K-Means Parameters */
typedef struct
{
	int numOfPoints;
	int numOfClusters;
	float timeLimit;
	float timeInterval;
	int iterationLimit;
	float qualityMeasure;
} K_MEANS_PARAMS;


/********************************************************************
*****					MACROS									*****
********************************************************************/
/* Patterns */
#define PATTERN_PARAMS											"%d %d %f %f %d %f"
#define PATTERN_POINTS											"%f %f %f %f %f %f"
#define PATTERN_RESULT_POINTS									"%f %f %f\n"
#define PATTERN_RESULT											"First Occurence Time = %f with Quality = %f\n"
#define PATTERN_CENTERS											"Centers of the Clusters:\n"
#define PATTERN_ALGO_FAILED										"Algorithm failed to calculate the quality before time/iteration limit reached.\n"

/* Termination Messages */
#define FAIL_ITERATION_LIMIT_REACHED							"Fail. Reached iteration limit.\n"
#define FAIL_TIME_LIMIT_REACHED									"Fail. Reached time limit.\n"
#define FAIL_UNSHIFTED_POINTS									"Fail. No points shifted between clusters.\n"
#define SUCCESS													"Success. Algorithm succeeded.\n"
#define FAIL_CUDA_FAILED										"Fail. Cuda failed to claculate clusters centers and diameters.\n"
#define CUDA_SINGLE_THREAD										1
#define CUDA_SINGLE_BLOCK										1

/* Consts */
#define NEW_LINE												"\n"
#define FILE_READ_MODE											"r"
#define FILE_WRITE_MODE											"w+"
#define FILE_PATH_INPUT											"C:\\input1.txt"
#define FILE_PATH_OUTPUT										"C:\\output.txt"
#define MASTER_PROCESS											0
#define NUM_OF_THREADS_OMP										10
#define NUM_OF_THREADS_PER_BLOCK_CUDA							500
#define X														0
#define Y														1
#define Z														2
#define DIRECTION_MASTER_SLAVE									0
#define DIRECTION_SLAVE_MASTER									1
#define TAG_STOP												0
#define TAG_OK													1

/* Functions */
#define PRINT_TERMINATION_CAUSE(a)								if(myId == MASTER_PROCESS) fprintf(stderr, a);
#define PRINT_TOO_MANY_PROC(a, b)								fprintf(stderr, "Process count must be lower than cluster count (P: %d - C: %d). Please run again with less processes.\n", a, b);

/********************************************************************
*****					PROTOTYPES								*****
********************************************************************/
/* K-Means */
BOOLEAN parallelKMeansExecution(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray, float* resultTime, float* resultQuality);
BOOLEAN parallelDistributePointsToClusters(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray);
BOOLEAN parallelRecalculateClusters(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray, CLUSTER* clusterArray);
float sequentialQualityCalculation(K_MEANS_PARAMS* parameters, CLUSTER* clusterArray);
BOOLEAN parallelRecalculatePointLocations(int myId, int worldSize, K_MEANS_PARAMS* parameters, POINT* pointArray);

/* Utils */
BOOLEAN syncProcesses(int myid, int worldSize, POINT* pointArray, CLUSTER* clusters, K_MEANS_PARAMS* parameters, BOOLEAN syncPoints, BOOLEAN syncClusters, int direction, int tag);
BOOLEAN readFile(K_MEANS_PARAMS* parameters, POINT** points, const char* fileName);
BOOLEAN writeFile(const K_MEANS_PARAMS* parameters, const CLUSTER* clusters, const float* resultTime, const float* resultQuality, const char* fileName);
BOOLEAN allocClusters(CLUSTER** clusters, int numOfClusters);
BOOLEAN allocPoints(POINT** points, int numOfPoints);
BOOLEAN initializeRandomCluster(CLUSTER* clusters, int numOfClusters);
BOOLEAN calcClusterStartAndEndIndices(int id, int worldSize, int numOfClusters, int* startClusterIndex, int* endClusterIndex);
BOOLEAN calcPointStartAndEndIndices(int id, int worldSize, int numOfPoints, int* startPointIndex, int* endPointIndex);

/* CUDA */
cudaError_t calcClusterCenterAndDiameterCuda(POINT* pointArray, CLUSTER* cluster, int pointArraySize);
cudaError_t allocMem(POINT** pointArrayCuda, CLUSTER** clusterCuda, const int pointArraySize, double** threadPointSumXYZ, double** threadMaxDiameter, int** threadNumOfPointsInCluster);
cudaError_t copyMem(POINT* pointArrayFrom, CLUSTER* clusterFrom, int pointArraySizeFrom, POINT* pointArrayTo, CLUSTER* clusterTo, cudaMemcpyKind direction);
__global__ void cudaSumPointPerThread(POINT* pointArrayCuda, CLUSTER* clusterCuda, int pointArraySizeCuda, double* threadPointSumXYZ, int* threadNumOfPointsInCluster);
__global__ void cudaCalcCenterTotal(CLUSTER* clusterCuda, double* threadPointSumXYZ, int* threadNumOfPointsInCluster);
__global__ void cudaCalcDiameterPerThread(POINT* pointArrayCuda, CLUSTER* clusterCuda, int pointArraySizeCuda, double* threadMaxDiameter);
__global__ void cudaCalcDiameterTotal(CLUSTER* clusterCuda, double* threadMaxDiameter, int threadCount);

/********************************************************************
*****					GLOBAL									*****
********************************************************************/

/********************************************************************
*****					DEBUG									*****
********************************************************************/
//#define __DEBUG_MODE
#define DEBUG_PRINT												printf("DEBUG LINE: %d FILE: %s\n", __LINE__, __FILE__);
#define DEBUG_PRINT_THREAD(a)									printf("DEBUG THREAD: %d LINE: %d FILE: %s\n", a, __LINE__, __FILE__);
#define DEBUG_PRINT_PROCESS(a)									if(a == 0) printf("DEBUG PROCESS: %d LINE: %d FILE: %s\n", a, __LINE__, __FILE__);
#define DEBUG_ITERATION_PRINT									printf("Proc: %d - Iteration: %d - Time: %f\n", myId, i, time);
#endif