package com.rikima.ml.logreg;

import com.rikima.ml.mlclassifier.mldata.FeatureVector;

public interface Minimizable {
    
    public FeatureVector minimize();
    public void init();
    public void reset(FeatureVector wv);
    
    public double[] w();
    public double[] grad();
}
