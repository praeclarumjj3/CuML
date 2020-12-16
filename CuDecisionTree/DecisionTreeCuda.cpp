//
// Created by Bhushan Pagariya on 6/1/17.
//

#include <iostream>
#include <vector>
#include <math.h>
#include <string>
#include <algorithm>
#include<math.h>
#include <omp.h>
#include<cstring>
#include<cstdlib>

#include<time.h>

#define GPU_TRIG_THRESHOLD 0
#define GPU_TRIG_SAMPLES_COUNT_THRESHOLD 0


using namespace std;

struct Node {
    int attributeId;
    float threshold;
    int classInd;
    float proportion;
    Node * leftChild;
    Node * rightChild;
    Node(int attributeId, float threshold){
        this->attributeId = attributeId;
        this->threshold = threshold;
        this->classInd = -1;
        this->proportion = -1;
        this->leftChild = NULL;
        this->rightChild = NULL;
    }
    Node(int attributeId, float threshold, int classInd, float proportion){
        this->attributeId = attributeId;
        this->threshold = threshold;
        this->classInd = classInd;
        this->proportion = proportion;
        this->leftChild = NULL;
        this->rightChild = NULL;
    }
};

void gpuHandler(float * samplesData, int * targetVals, int numSamples, int numAttrs, int numClasses, int * remAttributes, int & maxAttr, float & maxThreshold);
void alloc_cuda1D_float(int numSamples, int numAttrs);


float getThresholdSplit(float * data_mat, int numSamples, int numAttrs, int attribute){
    float max = 0;
    float min = 100000;
    for(int i = 0; i<numSamples; i++) {
        if(data_mat[i*numAttrs+attribute]>max)
            max = data_mat[i*numAttrs+attribute];
        if(data_mat[i*numAttrs+attribute]<min)
            min = data_mat[i*numAttrs+attribute];
    }    
    return (max+min)/2;
}

float computeEntropy(int * classHisto, int numClasses, int numSamples) {
    float entropy = 0;
    float proportion = 0;
    for(int i=0; i<numClasses; i++){
        //printf("class count-  %d ", classHisto[i]);
        if(classHisto[i] == 0)
            continue;
        proportion = ((float)classHisto[i])/numSamples;
        entropy += (-1)*proportion*log2(proportion);
    }
    return entropy;
}

void computeInfoGain(float * data_mat, int * targetValues, int start, int numSamples, int numAttrs, int attribute, float &informationGain, float &threshold, int numOfClasses){
    threshold = getThresholdSplit(data_mat+start*numAttrs, numSamples, numAttrs, attribute);
    //printf("Threshold for attr %d is %f\n",attribute, threshold);
    int parentClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        parentClassCount[i] = 0;
    int leftChildClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        leftChildClassCount[i] = 0;
    int rightChildClassCount[numOfClasses];
    for(int i = 0; i<numOfClasses; i++)
        rightChildClassCount[i] = 0;
    int leftSampleCount = 0;
    int rightSampleCount = 0;
    for(int i = start ; i<start+numSamples; i++) {
        if(data_mat[i*numAttrs+attribute] <= threshold) {
            leftChildClassCount[targetValues[i]]++;
            leftSampleCount++;
        } else {
            rightChildClassCount[targetValues[i]]++;
            rightSampleCount++;
        }
        parentClassCount[targetValues[i]]++;
    }
    if(leftSampleCount==0 || rightSampleCount==0){
        informationGain = 0;
        return;
    }


    float entropy_left = computeEntropy(leftChildClassCount, numOfClasses, leftSampleCount);
    float entropy_right = computeEntropy(rightChildClassCount, numOfClasses, rightSampleCount);
    float entropy_parent = computeEntropy(parentClassCount, numOfClasses, numSamples);
    //printf("Entropy left - %f, right - %f, parent - %f, numSamples - %d\n", entropy_left, entropy_right, entropy_parent, numSamples);
    float avgEntropy = (leftSampleCount/(numSamples))*entropy_left + (rightSampleCount/(numSamples))*entropy_right;
    informationGain = entropy_parent - avgEntropy;
}


Node * buildDecisionTree(float * dataMat, float * dataMat_bkp, int * targetVal, int * targetVal_bkp, int start, int numSamples, int numAttrs, int leafEntriesThreshold, int * remainingAttrs, int numOfClasses, int numThreads, int remAttrCount) {
    if(numSamples < leafEntriesThreshold) {
        // No information gain after splitting on any attribute
        int countPerClass[numOfClasses];
        // Initialize to 0
        for(int i = 0; i<numOfClasses; i++)
            countPerClass[i] = 0;

        for(int i = start; i<start+numSamples; i++)
            countPerClass[targetVal[i]]++;
        int max = 0;
        int index = -1;
        for(int i = 0; i<numOfClasses; i++)
            if(countPerClass[i]>max) {
                max = countPerClass[i];
                index = i;
            }

        return new Node(-1, -1, index, (float)max/numSamples);
    }

    float maxInfoGain=0, maxThreshold=0;
    int maxAttr = -1;
    // Compute Information Gain over all attributes
    float informationGain, threshold;
    int i;
    float infoGainArr[numAttrs];
    for(int j = 0; j<numAttrs; j++)
        infoGainArr[j] = -1;
    float thresholdArr[numAttrs];
    for(int j = 0; j<numAttrs; j++)
        thresholdArr[j] = -1;

//gpuHandler(dataMat+start*numAttrs, targetVal+start, numSamples, numAttrs, numOfClasses, remainingAttrs, maxAttr, maxThreshold);


    if(remAttrCount >= GPU_TRIG_THRESHOLD && numSamples>=GPU_TRIG_SAMPLES_COUNT_THRESHOLD) {
        gpuHandler(dataMat+start*numAttrs, targetVal+start, numSamples, numAttrs, numOfClasses, remainingAttrs, maxAttr, maxThreshold);
        //cout<<"In GPU"<<endl;
    } else {
        // in cpu
        for(i = 0; i< numAttrs; i++){
                if(remainingAttrs[i]!=0)
                    continue;
                computeInfoGain(dataMat, targetVal, start, numSamples, numAttrs, i, informationGain, threshold, numOfClasses);
                infoGainArr[i] = informationGain;
                thresholdArr[i] = threshold;
        }


        for(int i = 0; i<numAttrs; i++) {
            if(infoGainArr[i] > maxInfoGain) {
                maxInfoGain = infoGainArr[i];
                maxThreshold = thresholdArr[i];
                maxAttr = i;
            }
        }   
    }

    if(maxAttr == -1) {

        int countPerClass[numOfClasses];
        for(int i = 0; i<numOfClasses; i++)
            countPerClass[i] = 0;
        for(int i = 0; i<numSamples; i++)
            countPerClass[targetVal[i]]++;
        int max = 0;
        int index = -1;
        for(int i = 0; i<numOfClasses; i++)
            if(countPerClass[i]>max) {
                max = countPerClass[i];
                index = i;
            }

        return new Node(-1, -1, index, (float)max/numSamples);
    }

    remainingAttrs[maxAttr] = 1;
    int remForLeft[numAttrs], remForRight[numAttrs];
    memcpy(remForLeft, remainingAttrs, numAttrs*sizeof(int));
    memcpy(remForRight, remainingAttrs, numAttrs*sizeof(int));
    // Create Node
    Node * node = new Node(maxAttr, maxThreshold);

    //Split on attribute having max information gain
    
    int leftValueCount = 0;
    int rightValueCount = 0;
    for(int i=start; i<start+numSamples; i++) {
        if (dataMat[i*numAttrs+maxAttr] <= maxThreshold) {
            memcpy(dataMat+leftValueCount*numAttrs, dataMat+i*numAttrs, numAttrs*sizeof(float));
            targetVal[leftValueCount] = targetVal[i];
            leftValueCount++;
        } else {
            memcpy(dataMat_bkp+rightValueCount*numAttrs, dataMat+i*numAttrs, numAttrs*sizeof(float));
            targetVal_bkp[rightValueCount] = targetVal[i];
            rightValueCount++;
        }
    }
    //printf("Left Partition - %d, Right artition - %d\n", leftValueCount, rightValueCount);
    // merging both parts
    memcpy(dataMat+leftValueCount*numAttrs, dataMat_bkp, rightValueCount*numAttrs*sizeof(float));
    memcpy(targetVal+leftValueCount,targetVal_bkp,rightValueCount*sizeof(float));
    
    node->leftChild = buildDecisionTree(dataMat, dataMat_bkp, targetVal, targetVal_bkp, 0, leftValueCount, numAttrs, leafEntriesThreshold, remForLeft, numOfClasses, numThreads, remAttrCount-1);
    node->rightChild = buildDecisionTree(dataMat+leftValueCount*numAttrs, dataMat_bkp+leftValueCount*numAttrs, targetVal+leftValueCount, targetVal_bkp+leftValueCount, 0, rightValueCount, numAttrs, leafEntriesThreshold, remForRight, numOfClasses, numThreads, remAttrCount-1);

    return node;
}


void printDecisionTree(Node * root, string str, int & count) {
    if(!root)
        return;
    printDecisionTree(root->leftChild, str+"\t\t\t", count);
//    cout<<str<<"("<<root->attributeId<<","<<root->threshold<<","<<root->classInd<<","<<root->proportion<<")"<<endl;
    count++;
    printDecisionTree(root->rightChild, str+"\t\t\t", count);
}


void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  cout<<"Elapsed time: "<<elapsed<<" s\n";
}


int main(int argc, char** argv) {   
    int numSamples = 1000;
    int attrCount = 10;

    if(argc > 2) {
        numSamples = atoi (argv[1]);
        attrCount = atoi (argv[2]);
    }
   
    cout<<"Dataset size:- "<<numSamples<<"x"<<attrCount<<endl; 
    /* timer */
    clock_t start, stop;

    float * samples = (float *) malloc(numSamples*attrCount*sizeof(float));
    float * dataMat_bkp = (float *) malloc(numSamples*attrCount*sizeof(float));
    int offset = 0;
    for(int i = 0; i < numSamples; i++) {
        for(int j = 0; j<attrCount; j++) {
            samples[offset] = (float)(rand()%1000)/10;
            offset++;
        }
    }
    
    int * targetVals = (int *) malloc (numSamples*sizeof(int));
    int * targetVal_bkp = (int *) malloc (numSamples*sizeof(int));
    for(int i = 0; i<numSamples ; i++){
        targetVals[i] = rand()%10;
    }

    int * allAvailableAttrs = (int *) malloc (attrCount*sizeof(int));
    for(int i = 0; i<attrCount; i++)
        allAvailableAttrs[i] = 0;
    
    alloc_cuda1D_float(numSamples, attrCount);

    int remArr = 0;
    int maxAttr;
    float maxThreshold;
    gpuHandler(samples, targetVals, numSamples, 1, 10, &remArr, maxAttr, maxThreshold);

    /* initialize timer */
    start = clock();
    
    Node * rootNode = NULL;
    rootNode=buildDecisionTree(samples, dataMat_bkp, targetVals, targetVal_bkp, 0, numSamples, attrCount, 10, allAvailableAttrs, 10, 1000, attrCount);   
 
    stop = clock();
    print_elapsed(start, stop);

    int temp = 0;
    printDecisionTree(rootNode, "", temp);
    cout<<"Total nodes in tree:- "<<temp<<endl;
    return 0;
}
