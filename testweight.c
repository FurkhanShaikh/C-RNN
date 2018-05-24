#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define MAXWEIGHT ((float)0.3)		  //maximum weight
#define SCALEWEIGHT ((float)RAND_MAX)	//normalising scale factor
#define ITEMS 16			              //items prined per line

int vocab_size;

int get_vocab_size(char filename[]){
    FILE *ptr_file = fopen(filename,"r");
    char buf[1000];
    int distinct[256]={0};
    int i=0;
    int c;
    int count=0;
    if (!ptr_file){
      printf("Unable to read file\n");
      exit(1);
    }
    while (fgets(buf,1000, ptr_file)!=NULL){
      i=0;
      while(buf[i]!='\0'){
        c=buf[i];
        distinct[c]=1;
        i+=1;
      }
    }
    for(c = 0; c < 256; c++){
      if(distinct[c]){
        count += 1;
      }
    }
    fclose(ptr_file);
    return count;
}


void main(int argc, char *argv[]){

	float scale = SCALEWEIGHT; //normalising weights in range +- 1
	float wmax = MAXWEIGHT;		//wmax is abs val of max weight
  FILE *fpWeightsOut;

	int i,j,nInputNodes,nHiddenNodes,nOutputNodes;

  nHiddenNodes = 100;

	if(argc < 2){
		printf("Usage: testweights.exe inputfile [hiddensize]\n");
		exit(1);
	}
  srand(time(0)); //seed the random number generator
  char *input_file = argv[1];
  char *output_file = "initial_weights";
  if(argc == 3)
    nHiddenNodes = atoi(argv[2]); //number of hidden nodes
  vocab_size = get_vocab_size(input_file);
  nInputNodes = vocab_size; //number of input nodes
	nOutputNodes = vocab_size; // number of outpt nodes

  if((fpWeightsOut = fopen(output_file, "w"))== NULL){
    fprintf(stderr, "can't write file %s\n", output_file);
		exit(1);
  }

  //generate initial layer 1 weights, including a learnable bias for each hidden unit
  // printf("Initial Input:hidden weights with bias:\n");
  for(i=0;i<nHiddenNodes;i++){
		for(j=0;j<nInputNodes +1; j++){
			float frand = rand();
			float w1 = wmax *(1.0 - 2* frand/scale);
			// printf("%9.6f%c",w1,(j%ITEMS == (ITEMS -1) || j==nInputNodes) ? '\n':' ');
      fprintf(fpWeightsOut, "%9.6f%c",w1,(j%ITEMS == (ITEMS -1) || j==nInputNodes) ? '\n':' ');
		}
	}
	//generate initial hidden layer weights, including a learnable bias for each hidden unit
  // printf("\n\nInitial hidden to hidden wieghts with bias:\n\n" );
  for(i=0;i<nHiddenNodes;i++){
		for(j=0;j<nHiddenNodes +1; j++){
			float frand = rand();
			float w2 = wmax *(1.0 - 2* frand/scale);
			// printf("%9.6f%c",w2,(j%ITEMS == (ITEMS -1) || j==nHiddenNodes) ? '\n':' ');
      fprintf(fpWeightsOut, "%9.6f%c",w2,(j%ITEMS == (ITEMS -1) || j==nHiddenNodes) ? '\n':' ');
		}
	}
	//generate layer 2 weights with bias as above
  // printf("\n\ninital hidden to output weights with bias:\n\n" );
	for(i=0;i<nOutputNodes;i++){
		for(j=0;j<nHiddenNodes +1; j++){
			float frand = rand();
			float w3 = wmax *(1.0 - 2* frand/scale);
			// printf("%9.6f%c",w3,(j%ITEMS == (ITEMS -1) || j==nHiddenNodes) ? '\n':' ');
      fprintf(fpWeightsOut, "%9.6f%c",w3,(j%ITEMS == (ITEMS -1) || j==nHiddenNodes) ? '\n':' ');
		}
	}
  printf("Weights written to file %s\n",output_file );
}
