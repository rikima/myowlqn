package com.rikima.ml.mlclassifier.gic;

import java.util.*;
import java.io.*;

import cern.colt.matrix.*;
import cern.colt.matrix.impl.*;
import cern.colt.matrix.linalg.*;

import com.rikima.ml.logreg.Logreg;
import com.rikima.ml.mlclassifier.*;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.*;


public class MatrixGicCalculator {
    static PrintStream stderr = System.err;
    static PrintStream stdout = System.out;

    // fields ----------------------
    
    MLData mldata;
    FeatureVector weightVector;
    double c;
    double lambda;
    double n;
    int dim;

    DoubleMatrix2D R;
    DoubleMatrix2D RInverse;

    DoubleMatrix2D Q;

    // constructors -----------------

    /**
     * constructor
     * @param mldata
     * @param m
     */

    MatrixGicCalculator(MLData mldata, Model m) {
		
		this.mldata = mldata;
		this.weightVector = m.weightVector;
        
		this.n = mldata.size();
		this.dim = mldata.featureDimension();
		
		this.lambda = (double)m.c / n;
        
		stderr.println("#mldata.size()=" + n);
		stderr.println("#dim=" + dim);
		
		
		this.R = new DenseDoubleMatrix2D(dim, dim);
		this.Q = new DenseDoubleMatrix2D(dim, dim);
        
	}
	
	// methods --------------------
	
	/**
	 * calc loss
	 */
	public double calcLoss() {
		double loss = 0;
		for (int i = 0;i < mldata.size();++i) {
			CategoricalFeatureVector cfv = mldata.getCExample(i);
			
            double score = cfv.getClassValue() * cfv.dot(this.weightVector);
            loss += Math.log(1.0 + Math.exp(score));
		}
		
		return loss;
	}
	
	
	/**
	 * construct r matrix
	 */
	public void constructR() {
        stderr.print("constructing R ...");
		for (int i = 0;i < mldata.size();++i) {
            
			CategoricalFeatureVector cfv = mldata.getCExample(i);
			
			double score = - cfv.getClassValue() * cfv.dot(weightVector);
			double prob = Logreg.prob(cfv, weightVector);
			
			double coef = - prob * (1.0 - prob);
			
			for (int p = 0;p < cfv.size();++p) {
				
                int id_p = cfv.idByIndex(p);
                double x_p = cfv.valueByIndex(p);
				
                // diagonal
                double v = - (coef * x_p * x_p  - this.lambda);
                if (id_p == 1) {
                    v = - (coef * x_p * x_p);
                    
                }
                
                v += this.R.getQuick(id_p-1, id_p-1);
                this.R.setQuick(id_p-1, id_p-1, v);
                
                
                // non diagonal
                for (int q = p+1;q < cfv.size();++q) {
                    
                    int id_q = cfv.idByIndex(q);
                    double x_q = cfv.valueByIndex(q);
                    
                    
                    v = - coef * x_p * x_q;
                    v += this.R.getQuick(id_p-1, id_q-1);
                    this.R.setQuick(id_p-1, id_q-1, v );
                    this.R.setQuick(id_q-1, id_p-1, v);
                    
                }
            }
        }
        
        stderr.print(" .done!\n");
    }
    
    
    /**
     * construct q matrix
     */
    public void constructQ() {
        stderr.print("constructing Q ...");
        
        for (int i = 0;i < mldata.size();++i) {
            
            CategoricalFeatureVector cfv = mldata.getCExample(i);
            
            //double score = -cfv.getClassValue() * cfv.dot(this.weightVector); 
            double prob = Logreg.prob(cfv, this.weightVector);
            
            
            double coef = (1.0 - prob) * cfv.getClassValue();
            
            for (int j = 0;j < cfv.size();++j) {
                
                int id_j = cfv.idByIndex(j);
                double x_j = cfv.valueByIndex(j);
            	
                double psi_j = -coef * x_j - this.lambda * this.weightVector.valueById(id_j);
                if (id_j == 1) {
                    psi_j = -coef * x_j;
                }
                
                
                for (int k = 0;k < cfv.size();++k) {
                    
                    int id_k = cfv.idByIndex(k);
                    double x_k = cfv.valueByIndex(k);
                    
                    double v = psi_j * -coef * x_k;  
                    
                    v += Q.getQuick(id_j-1, id_k-1);
                    this.Q.setQuick(id_j-1, id_k-1, v);
                }
            }
        }
        stderr.print(" .done!\n");
        
    }
    
    
    
    
    public void calc() {
        constructR();
        constructQ();
        
        Algebra a = new Algebra();
        this.RInverse = a.inverse(this.R);
        
        double bias = a.trace(a.mult(RInverse, Q));
        double loss = calcLoss();
        
        stdout.println("# GIC=" + (loss+bias));
        stdout.println("# loss=" + loss);
        stdout.println("# bias=" + bias);
        
    }
    
    /**
     * main 
     * @param args
     */
    
    public static void main(String[] args) {
        String fname = null;
    	String model = null;
    	for (int i = 0;i < args.length;++i) { 
           if (args[i].equals("-i")) {
               fname = args[++i];
           }
           if (args[i].equals("-m")) {
        	   model = args[++i];
           }
       }
        if (model == null) {
            model = fname + ".model";
        }
        
        try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();
        
            Model m = LinearClassifier.readModel(model);
            
            
            MatrixGicCalculator self = new MatrixGicCalculator(mldata, m);
            self.calc();
        }
        catch (Exception e) {
        	e.printStackTrace();
        }
        
    }
	
    
}
