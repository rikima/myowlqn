package com.rikima.ml.logreg;

import java.util.Arrays;


//import com.justsystems.eureka.utils.ArrayUtils;
import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.*;

public class SemiSupervisedLogreg extends Logreg {
    static final boolean DEBUG = false;
    
    // fields ------------
    public final static double INITIAL_BETA = 1000;
    //private double beta = INITIAL_BETA;
    
    private static boolean useUnlabeled = false;
    
    private static boolean useUnlabeledLabel = false;
    
    private double prevLoss = Double.MAX_VALUE;
    
    public static final double LAMBDA = 1.0e-2;
    public double lambda = LAMBDA;
    
    // constructors ------
    /*
     * constructor
     * 
     */
    public SemiSupervisedLogreg(MLData mldata, double c, int type) {
		super(mldata, c, type);
    }

    /**
     * constructor
     * 
     * @param mldata
     * @param c
     * @param type
     * @param positiveWeight
     */
	public SemiSupervisedLogreg(MLData mldata, double c, int type, double positiveWeight) {
        super(mldata, c, type, positiveWeight);
    }
    
	// methods ----------------
    
	private void reset() {
        
		
	}
	
	
    
	/**
     * eval loss function
     * 
     */
    public double eval(double[] input, double[] gradient) {
        if (DEBUG) {
            stderr.println(this.getClass().getName() + "#eval()");
        }
        
		// initialize the gradient
		Arrays.fill(gradient, 0);
        
        double loss = 1.0;
        
        int missed = 0;
        for (int i = 0 ; i < mldata.size(); ++i) {
            CategoricalFeatureVector cfv = mldata.getCExample(i);
            
            if (DEBUG) {
                if (cfv.isUnlabeled()) { 
                    if (cfv.getClassValue() * cfv.y() < 0) {
                        stderr.println("Unlabeled:" + cfv.toString());
                        ++missed;
                    }
                }
            }
            
            if (!useUnlabeled && cfv.isUnlabeled()) {
            	continue;
            }
            
            double l_i = this.eachLoss(cfv, input, gradient);
            assert !Double.isNaN(l_i);
            loss += l_i; 
        }
        
        if (DEBUG) {
        	stderr.println("#missed=" + missed);
        }
        
        
        assert loss >= 0;
        if (DEBUG) {
            stderr.println("prevLoss=" + prevLoss + " loss=" + loss);
        }
        
        prevLoss = loss;
        return loss;
    }

	
	/**
	 *  calc loss value
	 * 
	 */
    private double eachLoss(CategoricalFeatureVector cfv, double[] input, double[] gradient) {
    	//score =  y * w * x
        double score = Double.NaN;
        if (cfv.isLabeled()) {
            score = cfv.y() * cfv.dot(input); 
        }
        else if (useUnlabeled && useUnlabeledLabel) {
            score = cfv.y() * cfv.dot(input); 
        }
    	else {
            score = cfv.dot(input);
    	}

    	// debug
        try {
            assert !Double.isNaN(score);
        }
        catch (AssertionError ae) {
        	ae.printStackTrace();
        }
        
        
        
        // test 
        if (DEBUG) {
        	if (cfv.isUnlabeled() && score == 0.0) {
            	stderr.println("score=" + score + " " + cfv.toString());
            }
            
            if (score  > 0) {
                stderr.print("#result y=+1");
            }
            else {
                stderr.print("#result y=-1");
            }
            stderr.println(" score=" + score + " " + cfv.toString());
        }
        
    	double insLoss, insProb;
        if (score < -30) {
			insLoss = - score;
            insProb = 0;
        } 
        else if (score > 30) {
			insLoss = 0;
			insProb = 1;
		} 
        else {
            double temp = 1.0 + Math.exp(-score);
            insLoss = Math.log(temp);
            insProb = 1.0 / temp;
		}
        
        
        double pp = Double.NaN;
        double np = Double.NaN;
        
        pp = insProb;
        np = 1.0 - pp;
        
        // supervised loss 
        if (cfv.isLabeled()) {
            if (cfv.getWeight() != 1.0) {
                assert cfv.getWeight() > 0;
                insLoss *= cfv.getWeight();
            }
        }
        //semi supervised loss or minimum entory
        else if (useUnlabeled){
            double e = 0;
            if (pp > 0) {
                e += pp * Math.log(pp);
            }
            
        	if (np > 0) {
        		e += np * Math.log(np);
        	}
        	
        	if (lambda != 1.0) {
        		e *= lambda;
        	}
        	
            if (useUnlabeledLabel) {
                insLoss += -e;
            }
            else {
                insLoss = -e;
            }
        }
            
        //double coef = Double.NaN;
        double coef = 0.0;
        // supervised case
        if (cfv.isLabeled()) {
            coef += - cfv.y() * (1.0 - insProb);
        }
        // unlabeled case
        else if (useUnlabeled){
            if (useUnlabeledLabel) {
                coef += - cfv.y() * (1.0 - insProb);
            }
            
        	assert cfv.isUnlabeled();
            if (pp * np > 0.0)  {
                coef += -pp * np * ( Math.log(pp) - Math.log(np) );
                
                if (lambda != 1.0) {
                	coef *= lambda;
                }
            }
        }
        
        // debug
        try {
            assert !Double.isNaN(coef);
        }
        catch (AssertionError e) {
            e.printStackTrace();
        }
        
        
        // update gradient
        for (int j = 0;j < cfv.size();++j) {
            double g = cfv.valueByIndex(j) * coef;
            
            assert !Double.isNaN(g);
            
            gradient[cfv.idByIndex(j)-1] += g;
        }	

        
        // set weight and class value for un labeled feature vectors
        if (cfv.isUnlabeled()) {
            if (pp >= 0.5) {
                cfv.setClassValue(CategoricalFeatureVector.POSITIVE);
                cfv.setProb(pp);
            }
            else {
                cfv.setClassValue(CategoricalFeatureVector.NEGATIVE);
                cfv.setProb(np);
            }
        }
        
        
        try {
            assert insLoss >= 0;
        }
        catch (Error err) {
            stderr.println("assertion error " + cfv.toString());
        }
            
        return insLoss;
    }
    
    /**
	 * main for test or exec
	 * @param args
	 */
	public static void main(String[] args) {
        
		if (args.length < 5) {
			stderr.println("please input -i [file name] -c [l1 weight] (-o [output model]) -l1 or -l2 -n2? -u [ration for unlabeled]");
            System.exit(1);
		}
        
        //Minimizable m = null;
        boolean l2normalize = false;
		int type = L1;
		
		String fname = null;
		String model = null;
		double c = 1.0;
		double pw = 1.0;
		double ratio = 0.0;
		
		boolean verbose = false;
        for (int i = 0;i < args.length;++i) {
			
			if (args[i].equals("-i")) {
                fname = args[++i];
			}
			else if (args[i].equals("-c")) {
				c = Double.parseDouble(args[++i]);
			}
			else if (args[i].equals("-m")) {
                model = args[++i];
            }
            else if (args[i].startsWith("-l")) {
                if (args[i].charAt(2) == '1') {
                	type = L1;
                }
                else if (args[i].charAt(2) == '2') {
                    type = L2;
                }
                    
			}
            else if (args[i].startsWith("-n2")) {
                l2normalize = true;
                stderr.println("# L2 normalization use.");
            }
            else if (args[i].startsWith("-pw")) {
                pw = Double.parseDouble(args[++i]);
                stderr.println("# positive weight=" + pw);
            }
            else if (args[i].startsWith("-v") || args[i].startsWith("--verbose")) {
                verbose = true;
            }
            else if (args[i].equals("-u")) {
                ratio = Double.parseDouble(args[++i]);
            }
            else if (args[i].startsWith("-ul") || args[i].startsWith("--unlabeled_label")) {
                useUnlabeledLabel = true;
            }
        }
		
        if (model == null) {
            model = fname + ".model.L" + type;
        }
        
		try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();
            
            if (pw != 1.0) {
            	mldata.setPositiveWeight(pw);
            }
                        
            if (l2normalize) {
                mldata.l2normalize();
            }
            
            
            SemiSupervisedLogreg self = new SemiSupervisedLogreg(mldata, c, type);
            SemiSupervisedLogreg.verbose = verbose;

            
            // supervised
            boolean retry = true;
            while (retry) {
                
                self.init();
                if (ratio > 0) {
                    mldata.setUnlabeled(ratio);
                    self.lambda = 1.0;
                }
                
                try {
                    self.useUnlabeled = false;
                    FeatureVector wv = self.train();
                    
                    // semi supervised 
                    self.useUnlabeled = true;
                    self.reset(wv);
                    wv = self.train();

                    self.outputWeightVector(model, wv);
                    retry = false;
                }
                catch (AssertionError ae) {
                    ae.printStackTrace();
                    retry = true;
                }
            }
        }
		catch (Exception e) {
			e.printStackTrace();
		}

	}
}
