package com.rikima.ml.mlclassifier.mldata.factory;


import java.io.*;
import java.util.*;

import com.rikima.ml.mlclassifier.mldata.MLData;


public class MLDataFactory {
    static final boolean DEBUG = false;

    // fields ------------------------------
    Loader loader;
    ArrayList<String> docs = new ArrayList<String>();

    public static boolean useBinaryFeature = true;

    // constructors ------------------------

    /**
     *  constructor
     */
    public MLDataFactory(String fileName) {
        this.loader = new SvmdataFormatLoader(fileName);
    }

    // method ------------------------------

    /**
     * return read mldata
     */
    public MLData get() throws Exception {
        return (MLData)loader.get();
    }

    public MLData getWithIndexor() throws Exception {
    	return (MLData)loader.getWithIndexor();
    }

    public MLData getWithIndexor(boolean saveDoc) throws Exception {
        MLData mldata = (MLData)loader.getWithIndexor(saveDoc);
        return mldata;
    }

    public void printFeatures() throws Exception {
    	SvmdataFormatLoader slr = (SvmdataFormatLoader)loader;

    	for (int i = 1;i < slr.featureIndexor.count();++i) {
            System.out.println(i + " " + slr.featureIndexor.getEntry(i));
        }
    }
}
