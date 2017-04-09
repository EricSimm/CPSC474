// author: Eric Simmons, Yijie Sun
// Project: Transpose a m*m dense matrix using multiple processes
// simulating a distributed system using docker and MPI
// input: text file with the number of row and col, the original matrix followed by the matrix body
// output: the transposed matrix

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char **argv) {

//declare variables
    int myrank, nprocs;
    int nrow, ncol;
    
    //open file and read the number of columns and rows
    FILE* fp;
    fp = fopen("matrix.txt", "r+");
    fscanf(fp,"%d %d", &nrow, &ncol);
    
    
    //create 2 dynamic arrays that are contigeous in memory
    int (*matrix)[nrow];
    int (*result)[ncol];
    if(myrank == 0){
        
        matrix = malloc(ncol * sizeof(*matrix));

        for(int i = 0; i<nrow; i++){
            for(int j = 0; j<ncol; j++){
                fscanf(fp,"%d", &matrix[i][j]);  
            }
        }
        
        result = malloc(nrow * sizeof(*result));

        
    }
    fclose(fp);
    //initalize MPI environment    
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);

    // If there are not enough processes exit the program
    if(nrow > nprocs)
    {
        MPI_Finalize();
        if(myrank ==0)
            printf("Number of processes need to be >= %d\n", nrow);
        exit(0);
    }
    
    //create new communicator. If number of processes is greater than the matrix size split the communicator into two groups
    //if the process = matrix size just set the new communicator = to MPI_COMM_WORLD
    MPI_Comm newworld;

    int color;
    if(nprocs > nrow){
        color = myrank<nrow?0:1;
        MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &newworld);
    }else{
        newworld = MPI_COMM_WORLD;
    }
    
    //reinitalize MPI environment
    int row_rank, row_size;
    MPI_Comm_rank(newworld, &row_rank);
    MPI_Comm_size(newworld, &row_size);



    //process 0 prints out the original matrix
    if(myrank == 0){
        printf("\n Ori: \n");
        for(int i =0; i< ncol; i++){
            for(int j=0;j<nrow; j++){
                printf("%d ", matrix[i][j]);
            }
            printf("\n");
        }
    }

    //create matrix buffers
    int local_row[ncol];
    int local_new_row[nrow];

    //MPI_Barrier(newworld);

    //scatter and reorganize the matrix
    MPI_Scatter(&matrix[row_rank], ncol, MPI_INT, &local_row, ncol, MPI_INT, 0, newworld);

    //MPI_Barrier(MPI_COMM_WORLD);
    /*printf("Rank: %d\n", myrank);
    for(int j = 0; j<ncol; j++){
        printf("%d ", local_row[j]);
    }
    printf("\n");*/
    free(matrix);

    MPI_Alltoall(&local_row, 1, MPI_INT, &local_new_row, 1, MPI_INT, newworld);

/*    
    printf("Rank: %d\n", myrank);
    for(int j = 0; j<ncol; j++){
        printf("%d ", local_new_row[j]);
    }*/

    MPI_Gather(&local_new_row, ncol, MPI_INT, &result[row_rank], ncol, MPI_INT, 0, newworld);

    //print out transposed matrix
    if(myrank == 0){
        printf("\n Result: \n");
        for(int i =0; i< ncol; i++){
            for(int j=0;j<nrow; j++){
                printf("%d ", result[i][j]);
            }
            printf("\n");
        }
    }
    
    if(nprocs > nrow){
        MPI_Comm_free(&newworld);
    }
    free(result);
    MPI_Finalize();
    
    


    return 0;
}
