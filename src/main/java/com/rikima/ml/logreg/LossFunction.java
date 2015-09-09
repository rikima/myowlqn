package com.rikima.ml.logreg;

/**
 * loss function interface
 * 
 * @author rikitoku
 *
 */
interface LossFunction {
    
	/**
	 * return loss function 
	 * 
	 * @param w weight vecotr 
	 * @param gradient gradinent to be updated.
	 * @return
	 */
    public double eval(double[] w, double[] gradient);
}
