package com.rikima.ml.mlclassifier.gic;


import java.io.*;
import java.util.Arrays;


import com.rikima.ml.logreg.*;
import com.rikima.ml.mlclassifier.*;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;


public class CvExecutor {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
    
    // fields --
    
    AbstractTrainer trainer;
    FeatureVector originalWeightVector;
    
    static double epsilon = 1.0e-4;
            
    double[] buf;
    
    // constructors ----
    
    /**
     * constructor
     * 
     */
    CvExecutor(AbstractTrainer t) {
    	this.trainer = t;
    }
    
    // methods ---------
    
    /**
     * set variational distributtion
     *
     *
     */
    protected void setCvDistribution(MLData mldata, int idx) {
    	for (int i = 0;i < mldata.size();++i) {
            double w = 1.0;
    		if (i == idx) {
                w = 0;
            }
    		mldata.getExample(i).setWeight(w);
        }
    }
    
    /**
     * noraml train
     * 
     */
    protected void normalTrain() {
        
        stderr.print("processing normali train...");
        // normal train
        try {
            FeatureVector wv = this.trainer.train();
            this.originalWeightVector = new FeatureVector(wv);
    	}
    	catch (Exception e) {
            e.printStackTrace();	
    	}
        stderr.println(" !done.\n");
    }
    
    
    /**
     * estimate gic
     * 
     * @param epsilon
     * @return
     */
    public double estimateCv() {
    	this.normalTrain();
        double lk = trainer.likellihood();
    	assert lk > 0;
        
    	stderr.println("#likellihood=" + lk);
    	
    	FeatureVector m = null;
    	
    	int size = trainer.getMLData().size();
    	
    	int pp = 0;
    	int pn = 0;
    	int np = 0;
    	int nn = 0;
    	
    	for (int i = 0;i < size;++i) {
            CategoricalFeatureVector cfv = trainer.getMLData().getCExample(i);
    		try {
                m = trainByVariationalDistribution(i);
    		}
    		catch (Exception e) {
    			e.printStackTrace();
    		}
            
            double y = m.dot(cfv) * cfv.getClassValue();
            
            if (cfv.getClassValue() > 0) {
                if (y > 0) {
                	pp++;
                }
                else {
                	pn++;
                }
            }
            else {
            	if (y > 0) {
            		np++;
            	}
            	else {
            		nn++;
            	}
            }
            
    	}
            	
        double acc = (double)(pp+nn)/size;
        stdout.println("# acc=" + acc + " (" + (pp+nn) + " + " + nn +  "/" + size + ")");
        stdout.println("# pp pn np nn=" + pp + " " + pn + " " + np + " " + nn);
        
        return acc;
    }
    
    /**
     * train by variational distribution
     * 
     * @param i
     * @return
     * @throws Exception
     */
    protected FeatureVector trainByVariationalDistribution(int i) throws Exception {
    	long t = System.currentTimeMillis();
    	stderr.println("-----");
    	stderr.println("#" + i + " variational train");
        
        MLData md = trainer.getMLData();
        
        setCvDistribution(md, i);
        trainer.reset(this.originalWeightVector);
        //trainer.init();
        
        FeatureVector wv = trainer.train();
        
        t = System.currentTimeMillis() - t;
        stderr.println("process time:" + t + " [ms]");
        return wv;
    }
    
    
        
    // main ----
	
    /**
	 * main 
	 * 
	 * @param args
	 */
	public static void main(String[] args) {
		double c = 1.0;
		String fname = null;
		
		int type = Logreg.L1;
		for (int i = 0;i < args.length;++i) {
            if (args[i].equals("-c")) {
                c = Double.parseDouble(args[++i]);
            }
			else if (args[i].equals("-i") || args[i].equals("--input")) {
                fname = args[++i];
			}
			else if (args[i].startsWith("-l")) {
                if (args[i].charAt(2) == '1') {
                    type = Logreg.L1;
                }
                else if (args[i].charAt(2) == '2') {
                    type = Logreg.L2;
				}
			}
		}
		
		if (fname == null) {
			stderr.println("please check options: -i [fname] -c [c value] -l1 or l2");
			System.exit(1);
		}
		
		try {
			
			MLDataFactory mf = new MLDataFactory(fname);
			MLData mldata = mf.getWithIndexor();
			
            Logreg trainer = new Logreg(mldata, c, type);

            CvExecutor self = new CvExecutor(trainer);
            self.estimateCv();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
 	}
	
	
}
