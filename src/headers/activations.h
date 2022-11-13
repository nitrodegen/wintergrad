/*
***********************************************************
    WinterGrad - awful ML library
        ML Activations module

            Should contain:
                Basic Activation functions like:
                    -Sigmoid
                    -Softmax
                    -ReLU
                    -LeakyReLU
                + addons
    
    @author:    nitrodegen
    @date:      Nov 2022.
***********************************************************           
*/
#ifndef WINTER_ACT_H
#define WINTER_ACT_H


#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "helpers.h"
#define e 2.718
using namespace std;


class WinterActivate{
    public:
        double vector_sum(vector<double>a){
            double res =0 ;
            for(int i =1;i<a.size();i++){
                res+=exp(a[i]);
            }
            return res;
        }
        double max(double a,double b){
            if(b> a){
                return b;
            }
            else{
                return a;

            }
        }
        double sigmoid(double num){
            return 1/(1+exp(-num));
        }
        vector<double> softmax_1d(vector<double> a){
            vector<double> res;
            double d= vector_sum(a);
            for(int i =0;i<a.size();i++){
                res.push_back((exp(a[i])/d));
            }
            return res;
        }
        double relu_1d(double num){
            return max(0,num);
        }
        void Softmax(Tensor1D *a ){
            vector<double> res = softmax_1d(a->tensor);
            a->tensor.clear();
            for(int i = 0;i<res.size();i++){
                a->tensor.push_back(res[i]);
            }
        }
        void ReLU1D(Tensor1D *a){
            vector<double>res;
            for(int i =0;i<a->tensor.size();i++){
                res.push_back(relu_1d(a->tensor[i]));
            }
            a->tensor.clear();
            for(int i =0;i<res.size();i++){
                a->tensor.push_back(res[i]);
            }         
        }

        double leakyrelu_1d(double curve,double num){
            return max(curve*num,num);
        }
        
};
#endif