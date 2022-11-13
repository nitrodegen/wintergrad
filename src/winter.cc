#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <time.h>
#include <math.h>
#include <unistd.h>
#include <algorithm>
#include "helpers.h"
#include "activations.h"
#include "layers.h"

using namespace std;

void perceptron_test(){

        WinterPresets *preset = new WinterPresets();
        vector<vector<double> >res = preset->IrisPerceptron(10,0,100);
        cout<<"\n"<<res[0][0]<<":"<<res[0][1]<<endl;

}

void mnist_predict(){
        //doing 2D shit and more complex predictions.

        dbg = true;//defined in helpers.
        WinterActivate* acts = new WinterActivate();
        WinterLayers* layer= new WinterLayers();
        WinterLoss* floss= new WinterLoss();            
        WinterPresets *presets = new WinterPresets();
        
        printf("\nhello from mnist,2D!\n");
        vector<vector<double> > mnist = presets->MNIST(true,NULL,1024);
        vector<vector<vector<double > > >  weights;
        int epochs = 100; 

        for(int g = 0;g<epochs;g++){
                //shuffling data...
               // mnist = presets->shuffle_3d_vector(mnist);
                for(int i=0;i<mnist.size();i++){

                        vector<double>  p = mnist[23];
                
                        Tensor1D *img = new Tensor1D(784);
                        img->fill_tensor(p);
                        Tensor1D *res = new Tensor1D(256);
                        Linear* first = new Linear(784,256,img,res,NULL);

                        acts->ReLU1D(res);
                        Tensor1D *fd= new Tensor1D(128);
                        Linear* sec = new Linear(256,128,res,fd,NULL);
                        acts->ReLU1D(fd);
                        Tensor1D *pred = new Tensor1D(10);
                        Linear* thr = new Linear(128,10,fd,pred,NULL);
                        acts->Softmax(pred);


                        Tensor1D *y = new Tensor1D(1);
                        y->tensor.push_back(presets->mnist_labels[i]);
                       
                        //TODO:IMPLEMENT CROSSENTROPYLOSS!
                        for(int j = 0;j<pred->tensor.size();j++){
                                cout<<pred->tensor[j]<<endl;

                        }


                        
                        exit(1);
                        //pushing all weights into one vector (for optimizer to optimize shit.)
                        
                        weights.push_back(first->w);
                        weights.push_back(sec->w);
                        weights.push_back(thr->w);
                        
                        //tensor cleanup 
                        img->tensor.clear();
                        res->free();
                        fd->free();
                        
                }
        }
        

}

int main(){

        mnist_predict();

}       