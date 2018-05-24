#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define ERRORLEVEL 0.20		//max allowed error
#define ITEMS 16			//items prined per line in file

typedef float *PFLOAT;
typedef PFLOAT VECTOR;
typedef PFLOAT *MATRIX;

void VectorAllocate(VECTOR *vector, int nCols);
void AllocateCols(PFLOAT matrix[], int nRows, int nCols);
void MatrixAllocate(MATRIX *pmatrix, int nRows, int nCols);
void MatrixFree(MATRIX matrix, int nRows);

static char* readcontent(char filename[], int*, int*);
int get_vocab_size(char filename[],int *nUnique);
void create_dict(int*);
float lossFun();

int vocab_size;
int *char_to_ix;
int *ix_to_char;
int *inputs;
int *targets;

float eta;					//default learning rate
int seq_length;        //no.of steps to unroll RNN
int nHiddenNodes;     //no of hidden nodes
int nInputNodes;
int nOutputNodes;

int i, j, n, p, t; 	// indexes
int npred;
int ix;

MATRIX W_input_hidden;	// weights input to hidden
MATRIX W_hidden_hidden;	// weights hidden to hidden
MATRIX W_hidden_output;	// weights hidden to output
MATRIX hprev;						// weights previous hidden state
MATRIX hpredict;				// weights of hidden state for sampling
MATRIX hptemp;					// temporary matrix to store weights of hidden state for sampling
MATRIX xs;							// input state
MATRIX hs;							// current hidden state
MATRIX ys;							// output state
MATRIX ps;							// output probabilities
MATRIX delwxh;					// change in weight input to hidden
MATRIX delwhh;					// change in weight hidden to hidden
MATRIX delwhy;					// change in weight hidden to output
MATRIX dhnext;					// delta current hidden to next
MATRIX dy;							// output gradient
MATRIX dh;							// hidden gradient
MATRIX dhraw;						// hidden gradient
MATRIX mWxh;						// memory variable for adagrad
MATRIX mWhh;						// memory variable for adagrad
MATRIX mWhy;						// memory variable for adagrad
MATRIX cdf;							// cumulative sum for non-uniform probability distribution

int main(int argc, char *argv[]){
	clock_t start, end;
  double cpu_time_used;
	start = clock();
	srand(time(0));									//seed random number generator

  float errorLevel = ERRORLEVEL;	// satisfactory error level
	register int hi;								// index hidden layer
	register int in;								// index input layer
	register int op;								// index output layer
  FILE *fpWeights;								// read initial weight file
	FILE *fpWeightsOut;

	int nReportErrors = 100;				// error reporting frequency
	int nIterations = 100000;				//total training steps
	eta = 0.15;
	nHiddenNodes = 100;
	seq_length = 25;
	npred = 200;
  // char input_file[] = "input.txt";	// input data file
	char weightsOutFile[] = "final_weights";

	if(argc < 2){
		printf("Usage: rnnet.exe inputfile [hiddensize]\n");
		exit(1);
	}

	char *input_file = argv[1];

	if(argc == 3)
		nHiddenNodes = atoi(argv[2]);

	char *data;
  int len_data;										// total characters
  int uniques[256];								// to find distinct characters
	for(i=0;i<256;i++)
    uniques[i]=0;

  inputs = (int*)malloc(seq_length*sizeof(int));	// input characters
  targets = (int*)malloc(seq_length*sizeof(int));	// target characters

  data = readcontent(input_file,&len_data, uniques);				// read the data file
  // vocab_size = get_vocab_size(input_file,uniques);
  nInputNodes = vocab_size;
  nOutputNodes = vocab_size;

  printf("data has %d characters, %d unique\n\n",len_data, vocab_size );

  char_to_ix = (int*)malloc(256*sizeof(int));					// char to index dict
  ix_to_char = (int*)malloc(vocab_size*sizeof(int));	// index to char dict
  create_dict(uniques);

  //-----------allocate storage for matrices------------------
  MatrixAllocate(&W_input_hidden,nHiddenNodes,nInputNodes + 1);  	//input to hidden wieghts + bias
  MatrixAllocate(&W_hidden_hidden,nHiddenNodes,nHiddenNodes + 1); //hidden to hidden weights + bias
  MatrixAllocate(&W_hidden_output,nOutputNodes,nHiddenNodes + 1); //hidden to output weights + bias
  MatrixAllocate(&delwxh, nHiddenNodes, nInputNodes + 1);
  MatrixAllocate(&delwhh, nHiddenNodes,nHiddenNodes + 1);
  MatrixAllocate(&delwhy, nOutputNodes,nHiddenNodes + 1);
  MatrixAllocate(&hprev, nHiddenNodes, 1);
  MatrixAllocate(&dhnext, nHiddenNodes,1);
  MatrixAllocate(&hpredict,nHiddenNodes,1);
	MatrixAllocate(&hptemp, nHiddenNodes, 1);
  MatrixAllocate(&xs, vocab_size, seq_length);
  MatrixAllocate(&hs,nHiddenNodes,seq_length);
  MatrixAllocate(&ys,nOutputNodes,seq_length);
  MatrixAllocate(&ps,nOutputNodes,seq_length);
  MatrixAllocate(&dy, nOutputNodes, 1);
  MatrixAllocate(&dh, nHiddenNodes, 1);
  MatrixAllocate(&dhraw,nHiddenNodes,1);
  MatrixAllocate(&mWxh,nHiddenNodes, nInputNodes +1);
  MatrixAllocate(&mWhh,nHiddenNodes,nHiddenNodes +1);
  MatrixAllocate(&mWhy,nOutputNodes,nHiddenNodes +1);
	MatrixAllocate(&cdf, nOutputNodes, 1);

  //----------Read the initial wieght matrices----------

  if((fpWeights = fopen("initial_weights", "r")) == NULL){
    printf("Can't open inital weights file\n");
    exit(1);
  }

  //read input:hidden weights
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nInputNodes +1; in++){
      fscanf(fpWeights,"%f",&W_input_hidden[hi][in]);
      mWxh[hi][in] = 0.0;
    }
  }

  //read hidden:hidden wieghts
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nHiddenNodes +1; in++){
      fscanf(fpWeights,"%f",&W_hidden_hidden[hi][in]);
			mWhh[hi][in] = 0.0;
    }
  }

  //read hidden:output wieghts
  for(op = 0; op < nOutputNodes; op++){
    for(hi = 0; hi < nHiddenNodes +1; hi++){
      fscanf(fpWeights,"%f",&W_hidden_output[op][hi]);
			mWhy[op][hi] = 0.0;
    }
  }
  fclose(fpWeights);

  //----------------Read in all paterns to be learned-----------
  n = 0;
  p = 0;
	float loss;
  float smooth_loss = -log(1.0/vocab_size)*seq_length;	// inital loss

  while(n<nIterations && smooth_loss > errorLevel){

    if((p + seq_length + 1 >= len_data) || n==0 ){
      for(hi=0;hi<nHiddenNodes;hi++){
        hprev[hi][0] = 0.0; //Reset RNN memory
      }
      p = 0;
    }

    //initialise inputs and targets
    for(i = 0; i < seq_length; i++){
      inputs[i] = char_to_ix[data[p + i]];
      targets[i] = char_to_ix[data[p + 1 + i]];
    }

    loss = lossFun();
    smooth_loss = smooth_loss * 0.999 + loss * 0.001;
    if(n%nReportErrors==0){

      //-----------------Sampling----------------
      for(hi = 0; hi < nHiddenNodes; hi++){
        hpredict[hi][0] = hprev[hi][0];
      }

      for(j = 0; j < vocab_size; j++){
        xs[j][0] = 0.0;
      }
      xs[inputs[0]][0] = 1;
			float sum = 0;
			float uniform_samples;
      for(t = 0; t < npred; t++){
        //compute new hidden state
        for(hi = 0; hi < nHiddenNodes; hi++){
          // sum = W_input_hidden[hi][nInputNodes]; //input to hidden bias
          sum = 0;
          // Matrix multiply input[i] and W_input_hidden[i]
          for(in = 0; in < nInputNodes; in++){
            sum += W_input_hidden[hi][in]*xs[in][0];
          }
          // Matrix multiply prev_hidden[i] and W_hidden_hidden[i]
          for(in = 0; in < nHiddenNodes; in++){
            sum += W_hidden_hidden[hi][in]*hpredict[in][0];
          }
          sum += W_hidden_hidden[hi][nHiddenNodes]; //hiden to hidden bias
          // hs[hi][t] =1.0/(1.0 + exp(-sum)); // use sigmoid
          hptemp[hi][0] = tanh(sum); // use tanh
        }
				for(hi = 0; hi < nHiddenNodes; hi++){
					hpredict[hi][0] = hptemp[hi][0];
				}
        //compute unnormalised log probabilities for next character
        for(op = 0; op < nOutputNodes; op++){
          sum = 0;
          for(hi =0; hi < nHiddenNodes; hi++){
            sum += W_hidden_output[op][hi]*hpredict[hi][0];
          }
          sum += W_hidden_output[op][nHiddenNodes]; //hidden to output bias
          ys[op][0] = sum;
        }

        //probabilities for next characters
        float expsum = 0;
        for(op = 0; op < nOutputNodes; op++){
          expsum += exp(ys[op][0]);
        }
        for(op = 0; op < nOutputNodes; op++){
          ps[op][0] = exp(ys[op][0])/expsum;
        }

        //find index using probability distribution

				sum = 0.0;
				for(op = 0; op < nOutputNodes; op++){
					sum += ps[op][0];
					cdf[op][0] = sum;
				}
				for(op = 0; op < nOutputNodes; op++){
					cdf[op][0] /= cdf[nOutputNodes-1][0];
				}
				uniform_samples = (double)rand()/ (double)RAND_MAX;
				//searchsorted
				ix= 0;
				for(op = nOutputNodes - 1; op > 0; op--){
					if(uniform_samples>= cdf[op-1][0] && uniform_samples <= cdf[op][0]){
						ix = op;
						break;
					}
				}
				for(j = 0; j < vocab_size; j++){
	        xs[j][0] = 0.0;
	      }
	      xs[ix][0] = 1;
				printf("%c",ix_to_char[ix] );

      }
			printf("\n--------------------------------------\n" );
      printf("Iteration %d, loss: %f\n",n, smooth_loss ); //print progress
			printf("======================================\n\n" );
    }

    p += seq_length;
    n += 1;
  }

	//--------Store Final weights---------------

	if((fpWeightsOut = fopen(weightsOutFile, "w")) == NULL){
		fprintf(stderr, "can't write file %s\n", weightsOutFile);
		exit(1);
	}
	// write input:hidden weights
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nInputNodes +1; in++){
      fprintf(fpWeightsOut,"%9.6f%c",W_input_hidden[hi][in],
			 (in%ITEMS == (ITEMS-1) || in == nInputNodes )? '\n':' ');
    }
  }

  //read hidden:hidden weights
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nHiddenNodes +1; in++){
			fprintf(fpWeightsOut,"%9.6f%c",W_hidden_hidden[hi][in],
			 (in%ITEMS == (ITEMS-1) || in == nHiddenNodes )? '\n':' ');
    }
  }

  //read hidden:output weights
  for(op = 0; op < nOutputNodes; op++){
    for(hi = 0; hi < nHiddenNodes +1; hi++){
			fprintf(fpWeightsOut,"%9.6f%c",W_hidden_output[op][hi],
			 (hi%ITEMS == (ITEMS-1) || hi == nHiddenNodes )? '\n':' ');
    }
  }
  fclose(fpWeightsOut);

	MatrixFree(W_input_hidden, nHiddenNodes);
	MatrixFree(W_hidden_hidden, nHiddenNodes);
	MatrixFree(W_hidden_output, nOutputNodes);
	MatrixFree(delwxh,nHiddenNodes);
	MatrixFree(delwhh,nHiddenNodes);
	MatrixFree(delwhy,nOutputNodes);
	MatrixFree(hprev, nHiddenNodes);
	MatrixFree(dhnext, nHiddenNodes);
	MatrixFree(hpredict, nHiddenNodes);
	MatrixFree(hptemp, nHiddenNodes);
	MatrixFree(xs, vocab_size);
	MatrixFree(hs, nHiddenNodes);
	MatrixFree(ys, nOutputNodes);
	MatrixFree(ps, nOutputNodes);
	MatrixFree(dy, nOutputNodes);
	MatrixFree(dh, nHiddenNodes);
	MatrixFree(dhraw, nHiddenNodes);
	MatrixFree(mWxh, nHiddenNodes);
	MatrixFree(mWhh, nHiddenNodes);
	MatrixFree(mWhy, nOutputNodes);
	MatrixFree(cdf, nOutputNodes);
	free(inputs);
	free(targets);
	free(char_to_ix);
	free(ix_to_char);

	end = clock();
	cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
	printf("\nTotal time: %f\n",cpu_time_used );
  return 0;
}

static char* readcontent(char filename[],int *len, int *nUnique){
    char *fcontent = NULL;
    int fsize = 0;
		int distinct[256]= {0};
		int c;
		int count = 0;
    FILE *fp;
    fp = fopen(filename, "r");
    if(fp) {
        fseek(fp, 0, SEEK_END);
        fsize = ftell(fp);
        rewind(fp);
        fcontent = (char*) malloc(sizeof(char) * fsize);
        fread(fcontent, 1, fsize, fp);
				j = 0;
				while ((c=fcontent[j])!='\0') {
	        distinct[c] = 1;
	        j += 1;
				}
				*len = j - 1;
				for(c = 0, i = 0; c < 256; c++){
		      if(distinct[c]){
		        count += 1;
		        nUnique[i] = c;
		        ++i;
		      }
		    }
				vocab_size = count;
        fclose(fp);
    }
		else{
			printf("Unable to read file\n");
			exit(1);
		}
    return fcontent;
}

void create_dict(int *uniques){
  int i=0;
  int c;
  for(i=0;i<vocab_size;i++){
    c=uniques[i];
    char_to_ix[c]=i;
    ix_to_char[i]=c;
  }
}

float lossFun(){
  register int hi;					//index hidden layer
	register int in;					//index input layer
	register int op;					//index output layer
  float loss = 0;
	float sum = 0;
	float expsum = 0;

  //-----------------------Forward Pass----------------------------

  // create one hot encoded vector (vocab_size x 1)
  for(t = 0; t < seq_length; t++){
    for(j = 0; j < vocab_size; j++){
      xs[j][t] = 0.0;
    }
    xs[inputs[t]][t] = 1.0;
  }

  for(t=0;t<seq_length;t++){
    //compute new hidden state
    for(hi = 0; hi < nHiddenNodes; hi++){
      // sum = W_input_hidden[hi][nInputNodes]; //input to hidden bias
      sum = 0;
      // Matrix multiply input[i] and W_input_hidden[i]
      for(in = 0; in < nInputNodes; in++){
        sum += W_input_hidden[hi][in]*xs[in][t];
      }
      // Matrix multiply prev_hidden[i] and W_hidden_hidden[i]
      if(t==0){
        for(in = 0; in < nHiddenNodes; in++){
          sum += W_hidden_hidden[hi][in]*hprev[in][0];
        }
      }
      else{
        for(in = 0; in < nHiddenNodes; in++){
          sum += W_hidden_hidden[hi][in]*hs[in][t-1];
        }
      }
      sum += W_hidden_hidden[hi][nHiddenNodes]; 	// hiden to hidden bias

      // hs[hi][t] =1.0/(1.0 + exp(-sum)); // use sigmoid
      hs[hi][t] = tanh(sum); // use tanh
    }

    // compute unnormalised log probabilities for next character
    for(op = 0; op < nOutputNodes; op++){
      sum = 0;
      for(hi = 0; hi < nHiddenNodes; hi++){
        sum += W_hidden_output[op][hi]*hs[hi][t];
      }
      sum += W_hidden_output[op][nHiddenNodes]; //hidden to output bias

      ys[op][t] = sum;
    }

    //probabilities for next characters
    expsum = 0;
    for(op = 0; op < nOutputNodes; op++){
      expsum += exp(ys[op][t]);
    }
    for(op = 0; op < nOutputNodes; op++){
      ps[op][t] = exp(ys[op][t])/expsum;
    }

    //calculate softmax ( cross-entropy loss) using negative log likelihood
    loss += -log(ps[targets[t]][t]);
  }

  //---------------BACKWARD PASS----------------------

  //----reset Deltas-----

  // input:hidden weight deltas
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nInputNodes + 1; in++){
      delwxh[hi][in] = 0.0; //delta for matrix
    }
  }

  // hidden:hidden weights deltas
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nHiddenNodes + 1; in++){
      delwhh[hi][in] = 0.0;
    }
  }

  // hidden:output weights deltas
  for(op = 0; op < nOutputNodes; op++){
    for(hi = 0; hi < nHiddenNodes + 1; hi++){
      delwhy[op][hi] = 0.0;
    }
  }

  for(hi=0;hi<nHiddenNodes;hi++){
    dhnext[hi][0] = 0.0;
  }


  // printf("=======BACKWARD PASS=======\n" );
  for(t = seq_length - 1; t >= 0; t--){
    for(op = 0; op < nOutputNodes; op++){
      dy[op][0] = ps[op][t];
    }
    dy[targets[t]][0] -= 1;

    // calculate delwhy = (vocab_sizex1)x(nHiddenNodesx1)T
		// =(vocab_size x 1)x(1 x nHiddenNodes)= (vocab_size x nHiddenNodes)
    for(op = 0; op < nOutputNodes; op++){
      for(hi = 0; hi < nHiddenNodes; hi++){
        delwhy[op][hi] += dy[op][0]*hs[hi][t];
      }
      delwhy[op][nHiddenNodes] += dy[op][0];  // derivative of output bias
    }

    // backprop into h
    // dh = (vocab_size x nHiddenNodes)T x (vocab_size x 1)
		// = (nHiddenNodes x vocab_size)x(vocab_size x 1)
		// = (nHiddenNodes x 1)
    for(hi = 0; hi < nHiddenNodes; hi++){
      sum =0;
      for(op = 0; op < nOutputNodes; op++){
          sum += W_hidden_output[op][hi]*dy[op][0];
      }
      dh[hi][0] = sum + dhnext[hi][0];
    }

    //backprop through tanh nonlinearity
    for(hi = 0; hi <nHiddenNodes; hi++){
      dhraw[hi][0] = (1 - hs[hi][t]*hs[hi][t]) * dh[hi][0];
      delwhh[hi][nHiddenNodes] += dhraw[hi][0];
    }
    // delwxh = (nHiddenNodes x 1)x(vocab_size x 1)T
		// =(nHiddenNodes x 1)x(1x vocab_size )= (nHiddenNodes x vocab_size)
    // delwhh = (nHiddenNodes x 1)x(nHiddenNodes x 1)T
		// =(nHiddenNodes x1)x(1x nHiddenNodes)
		// = (nHiddenNodes x nHiddenNodes)
    // dhnext = (nHiddenNodes x nHiddenNodes)T x(nHiddenNodes x1)
		// = (nHiddenNodesx1)
    for(hi = 0; hi < nHiddenNodes; hi++){
      for(in = 0; in < nInputNodes; in++){
        delwxh[hi][in] += dhraw[hi][0]*xs[in][t];
      }
      if(t==0){
        for(op = 0; op < nHiddenNodes; op++){
          delwhh[hi][op] += dhraw[hi][0]* hprev[op][0];
        }
      }
      else{
        for(op = 0; op < nHiddenNodes; op++){
          delwhh[hi][op] += dhraw[hi][0]* hs[op][t-1];
        }
      }
      sum = 0;
      for(op = 0; op < nHiddenNodes; op++){
        sum += W_hidden_hidden[op][hi]*dhraw[op][0];
      }
      dhnext[hi][0]=sum;
    }
  }

  //for exploding gradients
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nInputNodes +1; in++){
        if(W_input_hidden[hi][in]<-5){
          W_input_hidden[hi][in] = -5.0;
        }
        else if(W_input_hidden[hi][in] > 5){
          W_input_hidden[hi][in] = 5.0;
        }
    }
    for(in = 0; in < nHiddenNodes +1; in++){
      if(W_hidden_hidden[hi][in]<-5){
        W_hidden_hidden[hi][in] = -5.0;
      }
      else if(W_hidden_hidden[hi][in] > 5){
        W_hidden_hidden[hi][in] = 5.0;
      }
    }
  }
  for(op = 0; op <nOutputNodes; op++){
    for(hi = 0; hi < nHiddenNodes +1; hi++){
      if(W_hidden_output[op][hi]<-5){
        W_hidden_output[op][hi] = -5.0;
      }
      else if(W_hidden_output[op][hi] > 5){
        W_hidden_output[op][hi] = 5.0;
      }
    }
  }

  //update previous hidden
  for(hi = 0; hi < nHiddenNodes; hi++){
    hprev[hi][0] = hs[hi][seq_length-1];
  }

  //--------------------Updating weights-----------

  //perform parameter update with Adagrad
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nInputNodes + 1; in++){
      mWxh[hi][in] += delwxh[hi][in]*delwxh[hi][in];
      W_input_hidden[hi][in] += -eta * delwxh[hi][in] / sqrt(mWxh[hi][in] + 1e-8);
    }
  }
  for(hi = 0; hi < nHiddenNodes; hi++){
    for(in = 0; in < nHiddenNodes + 1; in++){
      mWhh[hi][in] += delwhh[hi][in]*delwhh[hi][in];
      W_hidden_hidden[hi][in] += -eta * delwhh[hi][in] / sqrt(mWhh[hi][in] + 1e-8);
    }
  }
  for(hi = 0; hi < nOutputNodes; hi++){
    for(in = 0; in < nHiddenNodes + 1; in++){
      mWhy[hi][in] += delwhy[hi][in]*delwhy[hi][in];
      W_hidden_output[hi][in] += -eta * delwhy[hi][in] / sqrt(mWhy[hi][in] + 1e-8);
    }
  }

  return loss;
}

//-----------Array storage allocaion routines----------
//Allocate space for vector of float cells for one dimensional dynamic vector[cols]
void VectorAllocate(VECTOR *vector, int nCols){
	if((*vector =(VECTOR)calloc(nCols,sizeof(float)))==NULL){
		fprintf(stderr,"Sorry! Not enough memory fo nodes\n");
		exit(1);
	}
}
//Allocate space for columns (float cells ) for dynamic two dimensional matrix[rows][cols]
void AllocateCols(PFLOAT matrix[],int nRows, int nCols){
	int i;
	for(i=0;i<nRows;i++){
		VectorAllocate(&matrix[i],nCols);
	}
}
//Allocate space for a two dimensional dynamic matrix[rows][cols]
void MatrixAllocate(MATRIX *pmatrix, int nRows, int nCols){
	if((*pmatrix =(MATRIX)calloc(nRows,sizeof(PFLOAT)))==NULL){
		fprintf(stderr,"Sorry! Not enough memory for nodes\n");
		exit(1);
	}
	AllocateCols(*pmatrix,nRows,nCols);
}
//free space for two dimensional dynamic array
void MatrixFree(MATRIX matrix, int nRows){
	int i;
	for(i=0;i<nRows;i++)
		free(matrix[i]);
	free(matrix);
}
