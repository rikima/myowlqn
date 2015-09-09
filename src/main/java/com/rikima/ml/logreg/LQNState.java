package com.rikima.ml.logreg;


import java.util.*;
import java.io.*;


import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.utils.ArrayUtils;

public class LQNState implements Minimizable {
    static boolean DEBUG = false;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;

    // fields -----------------
    
    static double C1 = 1e-8;
    static int MAX_ITER = 1000;
    static double TOL = 1.0e-8;
    
    // fields ------------------
    
    /** l1 weight */
    
    double C = 1.0;
    
    int dim = -1;
    int m = 10;

    boolean quiet = false;
    
    protected MLData mldata;
    
    private double[] alphas;
    
    private Stack<Double> roList;
    private Stack<double[]> sList;
    private Stack<double[]> yList;
    
    private Stack<Double> prevVals;
    
    private double[] dir;
    private double[] steepestDescDir;
    
    protected double[] grad;
    protected double[] w;

    protected LossFunction lossFunction;
    
    double[] newW;
    double[] newGrad;
    
    
    private int iter = 1; 
    private double value;
    
    // constructors ------------
    /**
     * constructor
     * 
     */
    public LQNState(LossFunction lossFunction, MLData mldata, double c) {
        this.mldata = mldata;
    	
    	this.C = c;
        
        
    	this.lossFunction = lossFunction;
    	this.dim = mldata.featureDimension();
        
    	init();
    }
    
    // methods -----------------
    
    public double[] w() {
    	return w;
    }
    
    public double[] grad() {
    	return grad;
    }
    
    /**
     * init
     */
    public void init() {
    	this.iter = 0;

    	this.alphas = new double[m];
    	
        this.dir = new double[dim+1];
        this.steepestDescDir = new double[dim+1];
        this.grad = new double[dim+1];
        this.w = new double[dim+1];
        
        this.newGrad = new double[dim+1];
        this.newW = new double[dim+1];
        
        lossFunction.eval(this.w, this.grad);
        
        this.sList = new Stack<double[]>();
        this.yList = new Stack<double[]>();
        this.roList = new Stack<Double>();
        
        this.value = evalL2();
        this.prevVals = new Stack<Double>();
        
    }
    
    /**
     * reset
     * @param wv
     */
    public void reset(FeatureVector wv) {
        this.iter = 0;
        
        Arrays.fill(dir, 0);
    	Arrays.fill(this.steepestDescDir, 0);
    	
    	Arrays.fill(w, 0);
    	Arrays.fill(newW, 0);
    	
    	Arrays.fill(grad, 0);
    	Arrays.fill(newGrad, 0);

        
        
        
        for (int i = 0;i < wv.size();++i) {
        	int fid = wv.idByIndex(i);
        	double v = wv.valueByIndex(i);
        	
            w[fid-1] = v;
        }
        lossFunction.eval(w, grad);
        
        Arrays.fill(alphas, 0);
        this.sList.clear();
        this.yList.clear();
        this.roList.clear();
    	
        this.prevVals.clear();
        this.value = evalL2();
    }
    
    /**
     * minimize
     * 
     */
    public FeatureVector minimize() {
        while (true) {
        	if (DEBUG) {
                stderr.println("#" + iter);
                stderr.println(" new_value=" + value);
        	}
        	
            long t = System.currentTimeMillis();
            updateDir();
            t = System.currentTimeMillis() - t;
            
            if (DEBUG) {
                stderr.println("update time: " + t + " [ms]");
            }
            
            t = System.currentTimeMillis();
    		backtrackingLineSearch();
            t = System.currentTimeMillis() - t;
            
            if (DEBUG) {
                stderr.println("line search time: " + t + " [ms]");
            }
            
    		if (convergence()) {
    			break;
    		}
    		
    		t = System.currentTimeMillis();
            shift();
            t = System.currentTimeMillis() - t;
            
            if (DEBUG) {
                stderr.println("shift time: " + t + " [ms]");
            }
        }
        
        // debug
        
        
        return createWeightVector();
    }
    
    
    
    
    /**
     * create weight vector
     * 
     * 
     * @param nonZeroCount
     * @return
     */
    private FeatureVector createWeightVector() {
        TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
        for (int i = 0;i < dim;++i) {
            if (Math.abs(w[i]) > 1.0e-10) {
                int fid = i+1;
                buf.put(fid, w[i]);
                //
                if (DEBUG) {
                    stderr.println(fid + " " + w[i]);
                }
            }
        }
        
        int c = 0;
        for (double v : w) {
            if (Math.abs(v) > 1.0e-10) {
            	++c;
            }
        }
        stderr.println("#non zero feature=" + c + "/" + dim);
        
        
        return new FeatureVector(0, buf);
    }
    
    /**
     * get value
     * 
     * @return
     */
    
    private double getValue() {
        
        double retVal = Double.MAX_VALUE;
        
        if (prevVals.size() > m/2) {
            
            double prevVal = prevVals.firstElement();
            if (prevVals.size() == m) {
                prevVals.remove(0);
            }
            
    	    double averageImprovement = (prevVal - this.value) / prevVals.size();
            double relAvgImpr = averageImprovement / Math.abs(this.value);
            
            retVal = relAvgImpr;
        } 
        else {
        	if (DEBUG) {
                stderr.println("  (wait for five iters) ");
            }
        }
        
        prevVals.add(this.value);
        return retVal;
    }
    
    /**
     * judge a convergence
     * 
     * @return
     */
    protected boolean convergence() {
        double termCritVal = getValue();
    	if (termCritVal < TOL) {
    		return true;
    	}
        
    	if (iter > MAX_ITER) {
    		return true;
    	}
    	
    	if (DEBUG) {
            stderr.println("#" + iter);
            stderr.println(" new_value=" + value);
    	}
    	
        return false;
    	
    }
    
    /**
     * update a direction
     * 
     */
    protected void updateDir() {
        makeSteepestDescDir();
    	mapDirByInverseHessian();
    }
    
    /**
     * calc steepest desc dir
     * 
     * eq (4) in AG and calc v_k
     * 
     */
    private void makeSteepestDescDir() {
    	
    	dir[0] = -grad[0];
        for (int i = 1; i < dim; ++i) {
            dir[i] = -grad[i] - C * w[i];
        }
    	        
    	ArrayUtils.copy(dir, steepestDescDir);
    }
    
        
    /**
     * calc corrected descent direction
     * 
     * 
     */
    private void mapDirByInverseHessian() {
        int m = sList.size();
        
        if (m != 0) {
        	
        	/*
        	 *  V_k-1 = (I - y_k-1 s_k-1^T/(s_k-1 * y_k-1) ?
        	 * 
             */
        	for (int i = m - 1;i >= 0;--i) {
                alphas[i] = -ArrayUtils.dot(sList.get(i), dir) / roList.get(i);
                addMult(dir, yList.get(i), alphas[i]);
        	}
        	
        	/**
        	 * diagonal elements ?
        	 */
        	
        	double[] lastY = yList.get(m - 1);
            double yDotY = ArrayUtils.dot(lastY, lastY);
    		double scalar = roList.get(m - 1) / yDotY;
            ArrayUtils.times(dir, scalar);

            
            /**
             * V_k-1^top ?
             * 
             */
    		for (int i = 0; i < m; ++i) {
    			double beta = ArrayUtils.dot(yList.get(i), dir) / roList.get(i);
    			addMult(dir, sList.get(i), -alphas[i] - beta);
    		}
        	
        }
    	
    }
    
    /**
     * eval loss + L2 reg
     * 
     * 
     * @return
     */
    protected double evalL2() {
        double val = lossFunction.eval(newW, newGrad);
        
        if (DEBUG) {
            stderr.println("loss:" + val);
        }
        
        if (C > 0) {
            for (int i=1; i < dim; ++i) {
                val += newW[i] * newW[i] * C * 0.5;
    		}
    	}

    	return val;
    }
    
    /**
     * get next point
     * 
     * @param alpha
     */
    
    private void getNextPoint(double alpha) {
    	addMultInto(newW, w, dir, alpha);
    }
    
    /*
     *  directional derivative
     * 
     */
    private double dirDeriv() {
        double val = 0.0;
        val = dir[0] * grad[0];
        for (int i = 1; i < dim; ++i) {
            if (dir[i] != 0) { 
                val += dir[i] * (grad[i] + C * w[i]);
            }
        }
        return val;
    }
    
    
    /**
     * line search
     * 
     */
    
    private void backtrackingLineSearch() {
    	double origDirDeriv = dirDeriv();
    	// if a non-descent direction is chosen, the line search will break anyway, so throw here
    	// The most likely reason for this is a bug in your function's gradient computation
    	if (origDirDeriv >= 0) {
    		if (DEBUG) {
                stderr.println("L-BFGS chose a non-descent direction: check your gradient!");
    		}
    		System.exit(1);
    	}
    	
    	double alpha = 1.0;
    	double backoff = 0.5;
    	if (iter == 1) {
            double normDir = Math.sqrt(ArrayUtils.dot(dir, dir));
    		alpha = (1 / normDir);
            backoff = 0.1;
            
            if (DEBUG) {
                stderr.println("alpha:" + alpha);
                stderr.println("normDir:" + normDir);
            }
    	}
        
        double oldValue = value;
        while (true) {
    		getNextPoint(alpha);
    		value = evalL2();
            
    		if (DEBUG) {
                stderr.println("alpha:" + alpha);
                stderr.println("value:" + value);
                
                stderr.println("oldValue:" + oldValue);
                stderr.println("C1:" + C1);
                stderr.println("alpha:" + alpha);
    		}
            
    		if (value <= (oldValue + C1 * origDirDeriv * alpha) ){
    			break;
    		}

            if (!quiet) {
                stderr.print(".");
            }

    		alpha *= backoff;
    	}

        if (!quiet) {
    		stderr.println();
    	}
    }
    
    
    /**
     * shift
     * 
     * store the s and y vectors.
     * 
     */
    private void shift() {
        double[] nextS = null;
        double[] nextY = null;
        
    	int listSize = sList.size();

    	if (listSize < m) {
            nextS = new double[dim+1];
            nextY = new double[dim+1];
    	}
    	
    	if (nextS == null) {
    		nextS = sList.firstElement();
    		sList.remove(0);
    		
    		nextY = yList.firstElement();
    		yList.remove(0);
    		roList.remove(0);
    	}

    	addMultInto(nextS, newW, w, -1);
    	addMultInto(nextY, newGrad, grad, -1);
    	
    	double ro = ArrayUtils.dot(nextS, nextY);

    	sList.add(nextS);
    	yList.add(nextY);
    	roList.add(ro);

        ArrayUtils.copy(newW, w);
        ArrayUtils.copy(newGrad, grad);
    	iter++;
    }
    
    /**
     * add multi, a[] += b[] * c
     * 
     * @param a
     * @param b
     * @param c
     */
    private void addMult(double[] a, double[] b, double c) {
		for (int i=0; i < a.length; i++) {
            a[i] += b[i] * c;
        }
    }
    
    /**
     * add mult into, a[] = b[] + c[] * d
     * 
     * @param a
     * @param b
     * @param c
     * @param d
     */
    private void addMultInto(double[] a, double[] b, double[] c, double d) {
        for (int i = 0; i < a.length; i++) {
    		a[i] = b[i] + c[i] * d;
    	}
    }
}
