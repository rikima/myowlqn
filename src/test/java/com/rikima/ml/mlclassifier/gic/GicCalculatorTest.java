package com.rikima.ml.mlclassifier.gic;

import java.io.PrintStream;

import com.rikima.ml.logreg.*;
import com.rikima.ml.mlclassifier.gic.GicCalculator;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;



import junit.framework.TestCase;

public class GicCalculatorTest extends TestCase {
    
	static String dataDir = "../mlclassifier_source/";
    static PrintStream stderr = System.err;
    
	static double eps = 1.0e2;
	
    
	public void testGicCalsByL1() throws Exception {
		
		String fname = "./data/test.svmdata.100";
		double c = 1.0;
        
        String[] args = {
            "-i", String.format("%s", fname), 
            "-c", String.format("%f", c),
            "-l1"
        };
        
        {
            GicCalculator.main(args);
            
        }
        
	}
	
    public void testGicCalsByL2() throws Exception {
		
		String fname = "./data/test.svmdata.100";
		double c = 1.0e2;
        
        String[] args = {
            "-i", String.format("%s", fname), 
            "-c", String.format("%f", c),
            "-l2"
            };
        
        {
            GicCalculator.main(args);
            
        }
        
	}
	
	/*
	public void testSetVariationalDist() throws Exception {
        String svmdata = dataDir + "w8a.train.svmdata";
        
        MLDataFactory mf = new MLDataFactory(svmdata);
        MLData mldata = mf.getWithIndexor();
        
        double c = 1.0;
        Logreg lr = new Logreg(mldata, c, Logreg.L1);
        assertEquals(45546, mldata.size());
        
        GicCalculator calc = new GicCalculator(lr);
        
        double e = 1.0e-3;
        // 0 
        {
            int idx = 0;
            calc.setVariationalDistribution(mldata, idx, e);
            
            
            
            for (int i = 0;i < mldata.size();++i) {
            	
            	double v = (1-e) ;
                
            	if (i == idx) {
                    v = (1-e) + mldata.size() * e;
                }
            	assertEquals(v, mldata.getCExample(i).getWeight(), eps);
            }
            
            
        }
        
        
        // 10 
        {
            int idx = 10;
            calc.setVariationalDistribution(mldata, idx, e);
            
            
            
            for (int i = 0;i < mldata.size();++i) {
            	
            	double v = (1-e) ;
                
            	if (i == idx) {
                    v = (1-e) + mldata.size() * e;
                }
            	assertEquals(v, mldata.getCExample(i).getWeight(), eps);
            }
            
            
        }
        
        
        
    }
    
	*/
}
