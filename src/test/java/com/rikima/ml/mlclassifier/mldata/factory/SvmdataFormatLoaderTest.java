package com.rikima.ml.mlclassifier.mldata.factory;

import junit.framework.TestCase;

import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.*;


public class SvmdataFormatLoaderTest extends TestCase {
    public void testLoadSecurityData() throws Exception {
        String fname = "/home/rikitoku/workspace/security_data/svmdata/unigram.tfidf/unigram.tfidf.svmformat";

        MLDataFactory mf = new MLDataFactory(fname);
        MLData mldata = mf.getWithIndexor();
        
        assertNotNull(mldata);
    }

    public void testTest() throws Exception {
        assertEquals(1, 1);
    }
}
