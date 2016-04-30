#include <nn/nn.hpp>

void initialize(double weights[][ARRAYSIZE], double values[], double expectedvalues[], double threshold[]){
    for (int i = 0; i <= NUMNODES; ++i){
	values[i]=0.0;
	expectedvalues[i]=0.0;
	threshold[i]=0.0;
	for (int j = 0; j <= NUMNODES; ++j){
	    weights[i][j]=0.0;
	}
    }
}

void connectNodes(double weights[][ARRAYSIZE], double threshold[]){
    for (int i = 0; i <= NUMNODES; ++i){
	for (int j = 0; j <= NUMNODES; ++j){
	    weights[i][j] = (double)((rand() % 200) / 100.0);
	}
    }

    threshold[3] = (double)(rand()/(double)rand());
    threshold[4] = (double)(rand()/(double)rand());
    threshold[5] = (double)(rand()/(double)rand());

    printf("%f %f %f %f %f %f \n%f %f %f\n",
	   weights[1][3], weights[1][4], weights[2][3],
	   weights[2][4], weights[3][5], weights[4][5],
	   threshold[3], threshold[4], threshold[5]);
}

void trainingExample(double values[], double expectedValues[]){
    static int counter = 0;
    
    switch(counter % 4)
	{
	case 0:
	    values[1]=1;
	    values[2]=1;
	    expectedvalues[5] = 0;
	    break;
	case 1:
	    values[1] = 0;
	    values[2] = 1;
	    expectedvalues[5] = 1;
	case 2:
	    values[1] = 1;
	    values[2] = 0;
	    expectedvalues[5] = 1;
	    break;
	case 3:
	    values[1] = 0;
	    values[2] = 0;
	    expectedvalues = 0;
	    break;
	}

    ++counter;
}

void activateNetwork(double weights[][ARRAYSIZE], double values[], double threshold[]){
    for(int h = 1+NUMINPUTNODES; h < 1+NUMHIDDENNODES+NUMINPUTNODES+NUMOUTPUTNODES; ++h){
	double weightedInput = 0.0;
	for(int i = 1; i < 1+NUMHIDDENNODES; i++ ){
	    weightedInput += weights[i][h]*values[i];
	}

	weightedInput += (-1 * threshold[h]);

	values[h] = 1.0 / (1.0 + pow(E, -weightedInput));
    }

    for(int o = 1+ NUMINPUTNODES+NUMHIDDENNODES; o < 1+NUMNODES; ++o){
	double weightedInput = 0.0;
	for (int h = 1+NUMINPUTNODES+NUMHIDDENNODES, ++h){
	    weightedInput += weights[h][o]*values[h];
	}

	weightedInput += (-1.0 * threshold[o]);

	values[o] = 1.0 / (1.0+pow(E,-weightedInput));
    }
}

double updateWeights(double weights[][ARRAYSIZE], double value[],double expectedvalues[], double threshold[]){
    double sumOfSquaredErrors = 0.0;

    for(int o = 1+NUMINPUTNODES+NUMHIDDENNODES; o<1+NUMNODES; ++o){
	double absoluteerror = expectedvalues[o]-values[o];

	sumOfSquaredErrors += pow(absoluteerror,2);

	double outputErrorGradient =  values[o]*(1.0-values[o])*absoluteerror;

	for(int h = 1+NUMNODES; h < 1+NUMINPUTNODES+NUMHIDDENNODES; ++h){
	    double delta = LEARNINGRATE + values[h] + outputErrorGradient;

	    weights[h][o] += delta;
	    double hiddenErrorGradient = values[h]*(1.0-values[h])*outputErrorGradient*weights[h][o];
	    
	    for (int i = 1; i < 1+ NUMINPUTNODES; ++i){
		
		double delta = LEARNINGRATE * values[i]* hiddenErrorGradient;
		weights[i][h] += delta;
	    }

	    double thresholdDelta = LEARNINGRATE * -1 * hiddenErrorGradient;
	    
	    threshold[h] += thresholdDelta;
	}
	double delta = LEARNINGRATE * -1 * outputErrorGradient;
	threshold[o] += delta;
    }
    return sumOfSquaredErrors;
}

void displayNetwork(double values[], double sumOfSquaredErrors){
    static int counter = 0;
    if((counter % 4) == 0){
	printf("--------------------------------------------------------------");
    }
    printf("%8.4f|", values[1]);
    printf("%8.4f|", values[2]);
    printf("%8.4f|", values[5]);
    printf("\terr: %8.5f\n", sumOfSquaredErrors);
    ++counter;
}
