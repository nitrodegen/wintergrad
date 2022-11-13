#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <time.h>
#define BIAS 1
#define EPOCHS 50
#define LR 0.01
const double e = 2.718281828459045; // idk why not #defined but sure 
int sigmoid(double num){
    return 1/1+pow(e,num);
}
double irisdata[100][5]={
    {5.1, 3.5, 1.4, 0.2, 0}, {4.9, 3.0, 1.4, 0.2, 0}, {4.7, 3.2, 1.3, 0.2, 0}, {4.6, 3.1, 1.5, 0.2, 0},{5.0, 3.6, 1.4, 0.2, 0},{5.4, 3.9, 1.7, 0.4, 0},{4.6, 3.4, 1.4, 0.3, 0},{5.0, 3.4, 1.5, 0.2, 0},{4.4, 2.9, 1.4, 0.2, 0},{4.9, 3.1, 1.5, 0.1, 0},
    {5.4, 3.7, 1.5, 0.2, 0},{4.8, 3.4, 1.6, 0.2, 0},{4.8, 3.0, 1.4, 0.1, 0},{4.3, 3.0, 1.1, 0.1, 0},{5.8, 4.0, 1.2, 0.2, 0},{5.7, 4.4, 1.5, 0.4, 0},
    {5.4, 3.9, 1.3, 0.4, 0},{5.1, 3.5, 1.4, 0.3, 0},{5.7, 3.8, 1.7, 0.3, 0},{5.1, 3.8, 1.5, 0.3, 0},{5.4, 3.4, 1.7, 0.2, 0},{5.1, 3.7, 1.5, 0.4, 0},
		{4.6, 3.6, 1.0, 0.2, 0},{5.1, 3.3, 1.7, 0.5, 0},{4.8, 3.4, 1.9, 0.2, 0},{5.0, 3.0, 1.6, 0.2, 0},{5.0, 3.4, 1.6, 0.4, 0},{5.2, 3.5, 1.5, 0.2, 0},
		{5.2, 3.4, 1.4, 0.2, 0},{4.7, 3.2, 1.6, 0.2, 0},{4.8, 3.1, 1.6, 0.2, 0},{5.4, 3.4, 1.5, 0.4, 0},{5.2, 4.1, 1.5, 0.1, 0},{5.5, 4.2, 1.4, 0.2, 0},
		{4.9, 3.1, 1.5, 0.1, 0},{5.0, 3.2, 1.2, 0.2, 0},{5.5, 3.5, 1.3, 0.2, 0},{4.9, 3.1, 1.5, 0.1, 0},{4.4, 3.0, 1.3, 0.2, 0},{5.1, 3.4, 1.5, 0.2, 0},
		{5.0, 3.5, 1.3, 0.3, 0},{4.5, 2.3, 1.3, 0.3, 0},{4.4, 3.2, 1.3, 0.2, 0},{5.0, 3.5, 1.6, 0.6, 0},{5.1, 3.8, 1.9, 0.4, 0},{4.8, 3.0, 1.4, 0.3, 0},{5.1, 3.8, 1.6, 0.2, 0},{4.6, 3.2, 1.4, 0.2, 0},{5.3, 3.7, 1.5, 0.2, 0},{5.0, 3.3, 1.4, 0.2, 0},
		{7.0, 3.2, 4.7, 1.4, 1},{6.4, 3.2, 4.5, 1.5, 1},{6.9, 3.1, 4.9, 1.5, 1},{5.5, 2.3, 4.0, 1.3, 1},
    {6.5, 2.8, 4.6, 1.5, 1},{5.7, 2.8, 4.5, 1.3, 1},{6.3, 3.3, 4.7, 1.6, 1},{4.9, 2.4, 3.3, 1.0, 1},{6.6, 2.9, 4.6, 1.3, 1},{5.2, 2.7, 3.9, 1.4, 1},
		{5.0, 2.0, 3.5, 1.0, 1},{5.9, 3.0, 4.2, 1.5, 1},{6.0, 2.2, 4.0, 1.0, 1},{6.1, 2.9, 4.7, 1.4, 1},{5.6, 2.9, 3.6, 1.3, 1},{6.7, 3.1, 4.4, 1.4, 1},
		{5.6, 3.0, 4.5, 1.5, 1},{5.8, 2.7, 4.1, 1.0, 1},{6.2, 2.2, 4.5, 1.5, 1},{5.6, 2.5, 3.9, 1.1, 1},{5.9, 3.2, 4.8, 1.8, 1},{6.1, 2.8, 4.0, 1.3, 1},
		{6.3, 2.5, 4.9, 1.5, 1},{6.1, 2.8, 4.7, 1.2, 1},{6.4, 2.9, 4.3, 1.3, 1},{6.6, 3.0, 4.4, 1.4, 1},{6.8, 2.8, 4.8, 1.4, 1},{6.7, 3.0, 5.0, 1.7, 1},
		{6.0, 2.9, 4.5, 1.5, 1},{5.7, 2.6, 3.5, 1.0, 1},{5.5, 2.4, 3.8, 1.1, 1},{5.5, 2.4, 3.7, 1.0, 1},{5.8, 2.7, 3.9, 1.2, 1},{6.0, 2.7, 5.1, 1.6, 1},
		{5.4, 3.0, 4.5, 1.5, 1},{6.0, 3.4, 4.5, 1.6, 1},{6.7, 3.1, 4.7, 1.5, 1},{6.3, 2.3, 4.4, 1.3, 1},{5.6, 3.0, 4.1, 1.3, 1},{5.5, 2.5, 4.0, 1.3, 1},
		{5.5, 2.6, 4.4, 1.2, 1},{6.1, 3.0, 4.6, 1.4, 1},{5.8, 2.6, 4.0, 1.2, 1},{5.0, 2.3, 3.3, 1.0, 1},{5.6, 2.7, 4.2, 1.3, 1},{5.7, 3.0, 4.2, 1.2, 1},{5.7, 2.9, 4.2, 1.3, 1},{6.2, 2.9, 4.3, 1.3, 1},{5.1, 2.5, 3.0, 1.1, 1},{5.7, 2.8, 4.1, 1.3, 1}
};

double weights[5];
void shuffledata (double irisdata[100][5]){
    srand(time(NULL));
    for(int i =0;i<95;i++){
        int randm = rand()%(95-1+0)+1;
        for(int j =0;j<5;j++){
            irisdata[i][j] = irisdata[randm][j];
        }
    }   
}
double dot(double arr[5]){
    double sum=0;
    for(int i =0;i<5;i++){
        sum+=arr[i]*weights[i];
    }
    return sum;
}

void randomize(){
    
  weights[0]=-0.035232;
  weights[1] = -0.136238;
  weights[2] = 0.241959;
  weights[3] =-0.389211;
  weights[4] =-0.474488;

}
int classify(double arr[5]){
    double dott  = dot(arr);
    //dott+=BIAS*weights[4];
    return sigmoid(dott);
}
void trainPerceptron(double arr[100][5]){
    for(int i =0;i<EPOCHS;i++){
        int x=0;
        int z =0;
   //    shuffledata(arr);
        for(int j =0;j<95;j++){
            
            double dotted = dot(arr[j]);
            //dotted+BIAS*weights[4];
            int act = sigmoid(dotted);
            double err = arr[j][4]-act;
            printf("\nweights: %lf %lf %lf %lf",weights[0],weights[1],weights[2],weights[3]);
             if(err!=0){
    
              x++;
                for(int k =0;k<4;k++){
                    weights[k]+=err*arr[j][k]*2;
                printf("\nAA:%lf",arr[j][k]);
        }
        //        weights[4]+=BIAS*err;
            }
            if(x ==0){
              
              break;
              }
        }
    }
}
int main(){
    srand(time(NULL));
    randomize();     
    printf("\n**** randomized weights:");
    for(int i =0;i<5;i++){
        printf("\nw:%lf",weights[i]);
    }

    trainPerceptron(irisdata);

    printf("\n**** trained weights:");
    for(int i =0;i<5;i++){
        printf("\nw:%lf",weights[i]);
    }
    shuffledata(irisdata);
    double testdata[10][5]={{5.4, 3.0, 4.5, 1.5, 1},{6.0, 3.4, 4.5, 1.6, 1},{6.7, 3.1, 4.7, 1.5, 1},{6.3, 2.3, 4.4, 1.3, 1},{5.6, 3.0, 4.1, 1.3, 1},{5.5, 2.5, 4.0, 1.3, 1},{5.5, 2.6, 4.4, 1.2, 1},{6.1, 3.0, 4.6, 1.4, 1},{5.8, 2.6, 4.0, 1.2, 1},{5.0, 2.3, 3.3, 1.0, 1}};
    
    for(int i =0;i<10;i++){
        int expect = testdata[i][4];
        int got  = classify(testdata[i]);
        printf("\nexpected -> %d \t got -> %d",expect,got);
    }

}