#pragma once

#include <cstdio>
#include <cstdlib>
#include <cmath>

const int NUMINPUTNODES = 2;
const int NUMHIDDENNODES = 2;
const int NUMOUTPUTNODES = 1;

const int NUMNODES = NUMINPUTNODES + NUMHIDDENNODES + NUMOUTPUTNODES;

const int ARRAYSIZE = NUMNODES + 1;
const int MAXITERATIONS = 131072;
const double E = 2.71828;
const double LEARNINGRATE = 0.2;

void initialize(double[][ARRAYSIZE], double[], double[], double[]);
void connectNodes(double[][ARRAYSIZE], double[]);
void trainingExample(double[], double[]);
void activateNetwork(double[][ARRAYSIZE], double[], double[]);
double updateWeights(double[][ARRAYSIZE], double[],double[],double[]);
void displayNetwork(double[], double);
