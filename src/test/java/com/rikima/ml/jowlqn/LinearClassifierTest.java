package com.rikima.ml.jowlqn;

import junit.framework.TestCase;

import java.io.PrintStream;

import org.json.*;


import com.rikima.ml.mlclassifier.LinearClassifier;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.*;
import com.rikima.ml.utils.*;


public class LinearClassifierTest extends TestCase {
    
	static PrintStream stderr = System.err;
	static PrintStream stdout = System.out;
	
	static String dataDir = "../mlclassifier_source/";
    static double eps = 1.0e-2;

    // methods 
    
    
    public void testScore() throws Exception {
    	String data = "w8a.train.svmdata";
        String fname = dataDir + data;
        String json = fname + ".model";
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData testdata = mf.getWithIndexor();
        
        JSONObject jo = new JSONObject(Reader.read(json));
        
        LinearClassifier cf = new LinearClassifier(testdata, jo);
        
        assertEquals(252, cf.getWeightVector().size());
        
        
        // 0 
        {
            FeatureVector fv = testdata.getCExample(0);
            
            double s1 = cf.score(fv);
            
            String str = "-1 41:1 54:1 117:1 250:1";
            double s2 = cf.score(str);
            
            stderr.println("fv:" + str);
            stderr.println("s1 " + s1);
            stderr.println("s2 " + s2);
            
            assertEquals(s1, s2, eps);
                    }
    	
        // 10 
        {
            FeatureVector fv = testdata.getCExample(1);
            String str = "-1 59:1 68:1 115:1"; 
            
            double s1 = cf.score(fv);
            double s2 = cf.score(str);
            
            stderr.println("fv:" + fv.toString());
            stderr.println("s1 " + s1);
            stderr.println("s2 " + s2);
            
            assertEquals(s1, s2, eps);
            
        }
        
    }
    
    
    public void testClassifyByClosedW8a() throws Exception {
        String data = "w8a.train.svmdata";
        String fname = dataDir + data;
        String json = fname + ".model";
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData testdata = mf.getWithIndexor();
        
        JSONObject jo = new JSONObject(Reader.read(json));
        
        LinearClassifier cf = new LinearClassifier(testdata, jo);
        
        assertEquals(252, cf.getWeightVector().size());
        
        
        cf.classify();
        cf.printResults();
    }
    
    
    public void testClassifyByClosedRealSim() throws Exception {
        
    	String data = "real-sim.svmdata";
        String fname = dataDir + data;
        String json = fname + ".model";
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData testdata = mf.getWithIndexor();
        
        JSONObject jo = new JSONObject(Reader.read(json));
        
        LinearClassifier cf = new LinearClassifier(testdata, jo);
        
        //assertEquals(2795, cf.getWeightVector().size());
        
        cf.classify();
        cf.printResults();
    }
    
    public void testClassifyByCloseA9a() throws Exception {
        
    	String data = "a9a.train.svmdata";
        String fname = dataDir + data;
        String json = fname + ".model";
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData testdata = mf.getWithIndexor();
        
        JSONObject jo = new JSONObject(Reader.read(json));
        
        LinearClassifier cf = new LinearClassifier(testdata, jo);
        
        assertEquals(105, cf.getWeightVector().size());
        
        
        cf.classify();
        cf.printResults();
    }
    
    
    public void testClassifyByCloseRcv1() throws Exception {
        
        String data = "rcv1_train.binary.svmdata";
        String fname = dataDir + data;
        String json = fname + ".model";
        
        MLDataFactory mf = new MLDataFactory(fname);
        MLData testdata = mf.getWithIndexor();
        
        JSONObject jo = new JSONObject(Reader.read(json));
        
        LinearClassifier cf = new LinearClassifier(testdata, jo);
        
        assertEquals(581, cf.getWeightVector().size());
        
        
        cf.classify();
        cf.printResults();
    }

    
    
}
