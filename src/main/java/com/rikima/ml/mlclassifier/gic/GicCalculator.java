package com.rikima.ml.mlclassifier.gic;


import java.io.*;
import java.util.Arrays;


import com.rikima.ml.logreg.*;
import com.rikima.ml.mlclassifier.*;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;


public class GicCalculator {
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
    GicCalculator(AbstractTrainer t) {
    	this.trainer = t;
    }
    
    // methods ---------
    
    /**
     * set variational distributtion
     *
     *
     */
    protected void setVariationalDistribution(MLData mldata, int idx,  double epsilon) {
    	for (int i = 0;i < mldata.size();++i) {
            double w = (1.0 - epsilon);
            if (i == idx) {
                w += epsilon * mldata.size();
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
    public double estimateBias(double epsilon) {
        this.normalTrain();
        double loss = trainer.likellihood();
    	assert loss > 0;
        
    	double bias = 0;
    	
    	FeatureVector m = null;
    	
    	int size = trainer.getMLData().size();
    	for (int i = 0;i < size;++i) {
    		try {
                m = trainByVariationalDistribution(i);
    		}
    		catch (Exception e) {
    			e.printStackTrace();
    		}
    		
    		double[] pdl = this.partialDeferentialLogLikelihood(i, trainer);
            
    		// ? why -
    		bias += -influenceFunction(m).dot(this.partialDeferentialLogLikelihood(i, trainer));
            
    	}
        
    	bias /= trainer.getMLData().size();
            	
        // 
        stdout.println("#  GIC=" + (loss+bias));
        stdout.println("# loss=" + loss);
        stdout.println("# bias=" + bias);
        
        return bias;
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
        
        setVariationalDistribution(md, i, epsilon);
        trainer.reset(this.originalWeightVector);
        //trainer.init();
        
        FeatureVector wv = trainer.train();
        
        t = System.currentTimeMillis() - t;
        stderr.println("process time:" + t + " [ms]");
        return wv;
    }
    
    /**
     * calc influence function
     * 
     * @param m
     * @param trainer
     * @return
     */
    FeatureVector influenceFunction(FeatureVector m) {
        try {
            m = m.plus(this.originalWeightVector, -1);
        	m.times(1.0 / epsilon);
    	}
        catch (Exception e) {
            e.printStackTrace();
        }
    	
        // debug
        stderr.println("l2 of m=" + m.l2norm());
        
        
        return m;
    }
    
    /**
     * 
     * @param idx
     * @param trainer
     * @return
     */
    double[] partialDeferentialLogLikelihood(int idx, AbstractTrainer trainer) {
        if (buf == null) {
        	buf = new double[trainer.featureDimension()];
        }
        Arrays.fill(buf, 0.0);
        
        trainer.getGradientElement(idx, this.originalWeightVector, buf);
        return buf;
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
			if (args[i].equals("-e") || args[i].equals("--epsilon")) {
                epsilon = Double.parseDouble(args[++i]);
			}
			else if (args[i].equals("-c")) {
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
			stderr.println("please check options: -i [fname] -c [c value] -e [epsilon value");
			System.exit(1);
		}
		
		try {
			
			MLDataFactory mf = new MLDataFactory(fname);
			MLData mldata = mf.getWithIndexor();
			
            Logreg trainer = new Logreg(mldata, c, type);

            GicCalculator self = new GicCalculator(trainer);
            self.estimateBias(epsilon);
		}
		catch (Exception e) {
			e.printStackTrace();
		}
		
 	}
	
	
}
