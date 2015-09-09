package com.rikima.ml.mlclassifier.mldata.factory;


import java.io.IOException;

public interface Loader {
    public void load() throws IOException;
    public Object get() throws Exception;
    public Object getWithIndexor() throws Exception;
    public Object getWithIndexor(boolean saveDoc) throws Exception;
}
