#ifndef HELPERS_H
#define HELPERS_H

#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <math.h>

#define ERRHDR "[ ERROR ]"
#define OKHDR "[ OK ]"
#define MISMATCH "[ MISMATCH ]"
#define WRNHDR "[ WARNING ]"
#define WINTER_FATAL 04

bool dbg; // used for debugging purpouses

using namespace std;

class Tensor1D{
    public:
     
        int size;
        vector<vector<double> > w;
        vector<double>tensor;
        Tensor1D(int ad){
            size = ad;
            if(dbg){
                printf("\n*** tensor %p initialized with size:%d",this,ad);
            }
        }
     
        void vect2d_to_1dtensor(vector<vector<double> >a){
            for(int i =0;i<a.size();i++){
                for(int j =0;j<a[0].size();j++){
                        tensor.push_back(a[i][j]);
                        
                }


            }
   
        }
        void fill_tensor( vector<double>  a ){
            if(a.size() !=size){
                printf("\n*** Tensor1D %p , with size shape [%d] of the vector, does not match the provided shape.",this,size);
                free();
                exit(WINTER_FATAL);

            }
            for(int i =0;i<a.size();i++){
               tensor.push_back(a[i]);
            }

        }
        void display(){
            for(int i =0;i<size;i++){
                if(size%4==0){
                    cout<<"\n"<<endl;
                }
                cout<<" "<<tensor[i];
                if(i <size-1){
                    cout<<",";
                }
            }
        }
        void free(){
            w.clear();
            tensor.clear();
            delete this;
        }
};

#endif