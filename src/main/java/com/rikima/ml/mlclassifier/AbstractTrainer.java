package com.rikima.ml.mlclassifier;

import java.io.*;

import org.json.*;

import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;

abstract public class AbstractTrainer {
    static boolean DEBUG = true;
    
    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;

    protected JSONObject model;
    
    // methods -------------------
    abstract public MLData getMLData();
    abstract public FeatureVector train() throws Exception;
    abstract public void reset(FeatureVector wv);
    abstract public void init();
    abstract public double[] getGradient();
    abstract public void getGradientElement(int idx, FeatureVector wv, double[] grad);
    abstract public double likellihood();
    abstract public double getC();
    
    /**
     * return dimension of feature space.
     * @return
     */
    public int featureDimension() {
    	return getMLData().featureDimension();
    }
    
    /**
     * outpu weight vector via json format
     * 
     * @throws IOException
     */
    public void outputWeightVector(String fname, FeatureVector weightVector) throws IOException {
        stderr.println("output weight vector to " + fname + " ...");
    	
        BufferedWriter bw 
            = new BufferedWriter( new OutputStreamWriter( new FileOutputStream(fname), System.getProperty("file.encoding")));
        
        assert getMLData().getIndex() != null;
        
        try {
            JSONObject jo = new JSONObject();
            jo.put("count", weightVector.size());
            jo.put("c", getC());
            
            JSONArray ja = new JSONArray();
            
            for (int i = 0;i < weightVector.size();++i) {
                
            	
            	
                String f = getMLData().getIndex().getEntry(weightVector.idByIndex(i));
                double v = weightVector.valueByIndex(i);
                
                
                try {
                    assert f != null;
                }
                catch (Error e) {
                	e.printStackTrace();
                    
                    stderr.println("i=" + i + " id=" + weightVector.idByIndex(i));
                    stderr.println("val=" + v);
                }
                    
                     
                // debug
                if (DEBUG) {
                    stderr.println(f + ":" + v);
                }
                
                JSONObject fjo = new JSONObject();
                fjo.put("rep", f);
                fjo.put("val", v);
                
                ja.put(fjo);
            }
            jo.put("weights", ja);
            
            bw.write(jo.toString());
            bw.close();
            
            stderr.println(" .done");
            
            this.model = jo;
        }
        catch (JSONException e) {
            e.printStackTrace();
        }
    }
}
