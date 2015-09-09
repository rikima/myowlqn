package com.rikima.ml.jowlqn;


import junit.framework.TestCase;

import java.io.*;


import com.rikima.ml.logreg.Logreg;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.*;



public class LogregTest extends TestCase {
    
	static String dataDir = "../mlclassifier_resources/";
    static PrintStream stderr = System.err;
    
	static double eps = 1.0e-2;
    
    public void testTrainByW8a() throws Exception {
        
        String svmdata = dataDir + "w8a.train.svmdata";
        
        MLDataFactory mf = new MLDataFactory(svmdata);
        MLData mldata = mf.getWithIndexor();
        double c = 1.0;
        Logreg lr = new Logreg(mldata, c, Logreg.L1);
        
        FeatureVector wv = lr.train();
        lr.outputWeightVector(svmdata + ".model", wv);
        
        // test 
        assertEquals(251, wv.size());
        
        {
            int idx = 0;
        	//stderr.println(idx + " " + lr.model.getJSONArray("weights").getJSONObject(idx).getString("rep"));
        	//assertEquals(-2.57496, lr.model.getJSONArray("weights").getJSONObject(idx).getDouble("val"), eps);
            
        }
         
        {
            int idx = 251;
            //stderr.println(idx + " " + lr.model.getJSONArray("weights").getJSONObject(idx).getString("rep"));
            //assertEquals(-2.57496, lr.model.getJSONArray("weights").getJSONObject(idx).getDouble("val"), eps);
            
        }
    }

    public void testTrainByRealSim() throws Exception {
        
        String svmdata = dataDir + "real-sim.svmdata";
        
        MLDataFactory mf = new MLDataFactory(svmdata);
        MLData mldata = mf.getWithIndexor();
        double c = 1.0;
        
        Logreg lr = new Logreg(mldata, c, Logreg.L1);
        
        FeatureVector wv = lr.train();
        lr.outputWeightVector(svmdata + ".model", wv);
        
        // test 
        
        // assertEquals(2795, wv.size());
        
        //assertEquals(-64.899, -wv.valueById(1), eps);
        //assertEquals(-1.7035, -wv.valueById(20936), eps);
        
    }
	
	
    public void testTrainByRcv1() throws Exception {
        
        String svmdata = dataDir + "rcv1_train.binary.svmdata";
        
        MLDataFactory mf = new MLDataFactory(svmdata);
        MLData mldata = mf.getWithIndexor();
        double c = 1.0;
        
        Logreg lr = new Logreg(mldata, c, Logreg.L1);
        
        FeatureVector wv = lr.train();
        lr.outputWeightVector(svmdata + ".model", wv);
        
        // test 
        
        assertEquals(580, wv.size());
        
         
        
	}
	
	
	
	
	public void testTrainByA9a() throws Exception {
        
        String svmdata = dataDir + "a9a.train.svmdata";
        
        MLDataFactory mf = new MLDataFactory(svmdata);
        MLData mldata = mf.getWithIndexor();
        double c = 1.0;
        
        Logreg lr = new Logreg(mldata, c, Logreg.L1);
        
        FeatureVector wv = lr.train();
        lr.outputWeightVector(svmdata + ".model", wv);
        
        // test 
        
        assertEquals(106, wv.size());
     
            
    }
    
	
	
	public void testTest() throws Exception {
		assertEquals(1, 1);
	}
	
}
