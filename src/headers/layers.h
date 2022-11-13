/*
***********************************************************
    WinterGrad - awful ML library
        ML Layers module

            Should contain:
                Convolutional Layer,
                Linear layer
                + addons
    
    @author:    nitrodegen
    @date:      Nov 2022.
***********************************************************           
*/
#ifndef LAYERS_H
#define LAYERS_H

#include <stdio.h>
#include <string.h>
#include <vector>
#include <stdlib.h>
#include <sstream>
#include <unistd.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "helpers.h"
#include <random>
#include <fstream>
#define DEFAULT_BIAS 1
#define use_bias(x) x/2.0
#define e 2.718
using namespace std;

vector<double> weights_2d(int size){
    vector<double> res;

    int filled =0; 
    random_device rd;
    double sized = 1./sqrt(size);
    mt19937 e2(rd());//  A Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
    uniform_real_distribution<> dist(-sized, sized);
    while(1){
        if(filled == size){
            break;
        }
        res.push_back(dist(e2));
        filled++;
    }   
    return res;
}




class Tensor2D{
    public:
     
        int x,y;
        vector<vector<double> > w;
        vector<vector<double> > tensor;

        Tensor2D(int xd,int yd){
            x=xd;
            y=yd;
            if(dbg){
                printf("\n*** Tensor2D %p initialized with size: [%d,%d]",this,x,y);
            }
        }
        void display(){
            for(int i =0;i<y;i++){
                printf("\n[");
                for(int j = 0;j<x;j++){
                    printf(" %lf, ",tensor[i][j]);

                }
                printf("]");
            }
        }
        void fill_tensor( vector<vector<double> >  a ){
            if(a.size() != y && a[0].size() != x){
                printf("\n*** Tensor2D %p , with size [%d,%d] shape of the vector, does not match the provided shape.",this,x,y);
                free();
                exit(WINTER_FATAL);

            }
            
            for(int i =0;i<a.size();i++){
                vector<double>d;
                for(int j = 0;j<a[0].size();j++){
                    d.push_back(a[i][j]);
                }
                tensor.push_back(d);
            }

        }
        void shape(){
            printf("\n[%d,%d]",x,y);
        }
        void free(){
            w.clear();
            tensor.clear();
            delete this;
        }
};
class WinterLoss{
    
    public:

        
        vector<double>BCELoss(Tensor1D *probability,Tensor1D *y){
            vector<double>res;
            /*
            //idk if this is right cause i might use it later to predict some shit with different sizes.

            if(a->tensor.size() != b->tensor.size()){
                printf("%s: a.size() is different than b.size()! (a.size() != b.size())",MISMATCH);
                exit(WINTER_FATAL);
            }
            */
            if(probability->tensor.size() > y->tensor.size()  ){
                printf("\n%s tensor1 size does not match the size of tensor2.\n",ERRHDR);
                exit(1);
            }
           
            int len = probability->tensor.size();
            for(int i =0;i<len;i++){
                double pyi = probability->tensor[i];
                double yi = y->tensor[i];
                double calc = -(1/len)*(yi*log(pyi)+(1-yi)*log(1-pyi));
                res.push_back(calc);
            }
        


            return res;
        }
};
 double dot1d(vector<double>a,vector<double>b){
    if(a.size() != b.size()){
        printf("%s: a.size() is different than b.size()! (a.size() != b.size())",MISMATCH);
        exit(WINTER_FATAL);
    }
    double r =0; 
    for(int i =0;i<a.size();i++){
        r+=(a[i]*b[i]);
    }
    return r;
}

class Linear{
        public:
            vector<vector<double > > w;
            void free(){
                w.clear();
                delete this;

            }
            Linear(int in,int out,Tensor1D *a,Tensor1D *ret , double *bias ){
                            
                            if(in != a->tensor.size()){
                                printf("\n%s input and tensor size do not match.",ERRHDR);
                                exit(WINTER_FATAL);
                            }
                            vector<double> res ;
                            //init_weights(in,out);
                            for(int i =0;i<out;i++){
                                vector<double>wvec = weights_2d(in);
                                if(wvec.size()==0){
                                    printf("\nexit");
                                    exit(1);
                                }
                                w.push_back(wvec);
                                                          
                            }
                          
                            
                            for(int i =0;i<out;i++){              
                        
                                if(bias == NULL){
                                    double resz = dot1d(a->tensor,w[i]);
                                    
                                    res.push_back(resz);
                                }
                                else{
                                    double ddd = dot1d(a->tensor,w[i]);
                                    ddd+=bias[0];
                                    res.push_back(ddd); 
                                }
                            
                            }
                         
                            
                            ret->tensor.clear();
                            for(int i =0;i<res.size();i++){
                                ret->tensor.push_back(res[i]);
                            }
                            //a->tensor = res;
                            a->size = out;
                            

        
            }
};

class WinterLayers{
    public:

        
        void init(){
            srand(time(NULL));
            if(dbg){
                printf("\n%s WinterLayers->init() initialized.\n",OKHDR);

            }
        }

        void linear_1d(int in,int out,Tensor1D *a,Tensor1D *ret , double *bias){
            if(in != a->tensor.size()){
                printf("\n%s input and tensor size do not match.",ERRHDR);
                exit(WINTER_FATAL);
            }
            vector<double> res ;
            int len = a->w.size();
            for(int i =0;i<out-len;i++){
                vector<double>d = weights_2d(in);
                a->w.push_back(d);
            }
            
            for(int i =0;i<out;i++){              
           
                if(bias == NULL){
                    double resz = dot1d(a->tensor,a->w[i]);
                    res.push_back(resz);
                }
                else{
                    double ddd = dot1d(a->tensor,a->w[i]);
                    ddd+=bias[0];
                    res.push_back(ddd); 
                }
            
            }
            //exit(1);
            for(int i =0;i<res.size();i++){
                ret->tensor.push_back(res[i]);
            }
            //a->tensor = res;

            a->size = out;
            
       }   


};

vector<string> split1(string str,string del){
	vector<string>dele;
	ssize_t beg,pos=0;
	while((beg=str.find_first_not_of(del,pos)) != string::npos){ // loop until you find everything that isn't a delimiter , and always set that to be beginning
		pos = str.find_first_of(del,pos+1);//position is always the next case of del
	
		dele.push_back(str.substr(beg,pos-beg)); // and push 
	}
	return dele;
}

class WinterPresets{

    public:

        vector<int> mnist_labels;

        vector<vector<vector<double> > > shuffle_3d_vector(vector<vector<vector<double> > >  a){
            vector<vector<vector<double> > > res;
            srand(time(NULL));
            int len = a.size();
            for(int i = 0;i<a.size();i++)   {
                int img_num =(rand()%(28-2))+1;
                res.push_back(a[img_num]);
            }
            return res;

        }
        vector<vector<double> >   MNIST(bool cached,char* filename,int size){
            vector<vector<double>  > ret;
            ifstream strm("./src/presets/mnist.csv");

            stringstream ss;
            ss<<strm.rdbuf();
            
            string dd(ss.str());
           
            vector<string> images = split1(dd,"\n");

            if(size > images.size()){
                printf("\n*** MNIST: not enough examples loaded.");
                exit(WINTER_FATAL);
            }
            for(int i =1;i<size;i++){   
                vector<string>img = split1(images[i],",");
                vector< double >pixels;
                int chk = 0;
                       
                for(int j= 1;j<img.size();j++){
                  
                    
                    double pixie=stoi(img[j]);
                    pixie = (double)pixie/255.0;
                    pixels.push_back(pixie);
                    

                }
                double label = stoi(img[0]);
                ret.push_back(pixels);
                mnist_labels.push_back(label);

            }

            if(dbg){
                printf("\n*** MNIST dataset loaded, examples:%d",size);
            }
   


    
            return ret;
        }
        vector<vector<double> >  IrisPerceptron(int preds,int ebg,int epochs_size){
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

            dbg = ebg;//defined in helpers.h
            WinterActivate* acts = new WinterActivate();
            WinterLayers* layer= new WinterLayers();
            WinterLoss* floss= new WinterLoss();
            

            int in = 4;
            int out = 1;

            layer->init();
            Tensor1D *t = new Tensor1D(in);
            Tensor1D *yi = new Tensor1D(out);
            int epochs = 90; 
            double wd[5];
            
            for(int g =0 ;g<epochs_size;g++){
                int x = 0; 
                for(int i =0;i<90;i++){

                        //pushing iris data to tensors...
                        for(int j =0;j<in;j++){
                                t->tensor.push_back(irisdata[i][j]);
                        }
                        yi->tensor.push_back(irisdata[i][4]);   
                        //creating a temp tensor to hold our return value.

                        Tensor1D *ret = new Tensor1D(out);
                        layer->linear_1d(in,out,t,ret,NULL);         
                        double val = ret->tensor[0];
                        val = acts->sigmoid(val);
                        ret->tensor.clear();
                        ret->tensor.push_back( val);
                    
                        vector<double>loss;
                        loss.push_back(yi->tensor[0]-ret->tensor[0]);
                        if(dbg){
                            printf("\nlinear_1d(): ret_val:%lf , real:%lf, tensor size:%ld\n",val,yi->tensor[0],ret->tensor.size());                
                            printf("\nlinear_1d():weights len:%d, loss:%lf",t->w[0].size(),loss[0]);
                            cout<<"\nlinear_1d():weights:"<<t->w[0][0]<<":"<<t->w[0][1]<<":"<<t->w[0][2]<<":"<<t->w[0][3]<<endl;
                            cout<<"\nlinear_1d():x:"<<x<<endl;
                        }
                        double ls = loss[0];
                        if(loss[0] !=0){
                                x++;
                                for(int k = 0 ;k<4;k++){
                                        t->w[0][k]+=ls*irisdata[i][k]*2;
                                        wd[k]= t->w[0][k];
                                }
                        }
                        if(dbg){                      
                         cout<<"\nlinear_1d(): weights after:"<<t->w[0][0]<<":"<<t->w[0][1]<<":"<<t->w[0][2]<<":"<<t->w[0][3]<<endl;
                        }

                        //clearing the previous data of our tensors...               
                        t->tensor.clear();
                        yi->tensor.clear();
                        //exit(1);
                }
                cout<<"\n"<<OKHDR<<" Training.. Epoch:"<<g;
            }
            double testdata[10][5]={{5.4, 3.0, 4.5, 1.5, 1},{6.0, 3.4, 4.5, 1.6, 1},{6.7, 3.1, 4.7, 1.5, 1},{6.3, 2.3, 4.4, 1.3, 1    },{5.6, 3.0, 4.1, 1.3, 1},{5.5, 2.5, 4.0, 1.3, 1},{5.5, 2.6, 4.4, 1.2, 1},{6.1, 3.0, 4.6, 1.4, 1},{5.8, 2.6, 4.0, 1.2, 1},    {5.0, 2.3, 3.3, 1.0, 1}};
    
            vector<vector<double> >prediz;;
            cout<<"\n**** Training done... ****"<<endl;
            Tensor1D *b = new Tensor1D(1);
            for(int i =0 ;i <preds;i++){
                    b->tensor.push_back(testdata[i][4]);               
                    for(int j =0;j<4;j++){
                            t->tensor.push_back(testdata[i][j]);
                    }
                    Tensor1D *ret = new Tensor1D(out);
                    layer->linear_1d(in,out,t,ret,NULL);         
                    double val = ret->tensor[0];
                    val = acts->sigmoid(val);
                    vector<double>dod;
                    dod.push_back(val);
                    dod.push_back(b->tensor[0]);
                    prediz.push_back(dod);

                    b->tensor.clear();
                    t->tensor.clear();

            }
            cout<<endl;
            return prediz;
        }
};
#endif 