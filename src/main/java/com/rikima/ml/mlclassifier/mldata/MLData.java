/*
 * 作成日: 2005/11/28
 *
 * TODO この生成されたファイルのテンプレートを変更するには次へジャンプ:
 * ウィンドウ - 設定 - Java - コード・スタイル - コード・テンプレート
 */
package com.rikima.ml.mlclassifier.mldata;

import java.io.*;

import java.util.*;

import com.rikima.ml.utils.*;

public class MLData implements Iterator, Serializable {
    
    // fields ---------------------------------
    protected Indexor indexor;
	protected int featureDimension;
    
	public PrintStream stdout = System.out;
    public PrintStream stderr = System.err;
    
    
    private ArrayList<FeatureVector> featureVectors;
    
    private int numPositive = -1;
    private int numNegative = -1;
    private int numUnlabeled = -1;
    
    private int size = -1;
    
    private int _ptr;
    
    // constructors ---------------------------
    
	/**
	 * constructor
	 */
    public MLData() {
        this.featureVectors = new ArrayList<FeatureVector>();
    }
    
    
	// methods --------------------------------
	
    public void setUnlabeled(double ratio) {
    	assert ratio > 0.0 && ratio < 1.0;
    	
    	int ul = (int)(this.featureVectors.size() * ratio);
        int[] rindices = ArrayUtils.randomRange(size());
    	
        // set all labeld for init
        for (int i = 0;i < size();++i) {
            CategoricalFeatureVector cfv = this.getCExample(i);
            cfv.setLabeled();
            assert cfv.isLabeled();
        }
    	
        for (int i = 0;i < ul;++i) {
            int ridx = rindices[i];
    		CategoricalFeatureVector cfv = this.getCExample(ridx);
    		
    		cfv.setGussianRandomClassValue();
            cfv.setUnlabeled();
            assert cfv.isUnlabeled();
        }
    	
    	
    	
    	// assert check
    	int pc = positiveSize();
    	int nc = negativeSize();
        int uc = unlabeledSize();
        
    	stdout.println("#positive=" + pc);
    	stdout.println("#negative=" + nc);
    	stdout.println("#unlabeled=" + uc);
    	
    	
    	
    	assert pc > 0 && nc > 0;
    }
    
    public void setPositiveWeight(double pw) {
    	for (int i = 0; i < this.featureVectors.size();++i) {
            CategoricalFeatureVector cfv = this.getCExample(i);
            if (cfv.classValue > 0) {
                cfv.setWeight(pw);
            }
    	}
    }
    
    
    public void l2normalize(){
        for (int i = 0; i < this.featureVectors.size();++i) {
            FeatureVector fv = this.featureVectors.get(i);
            fv.l2normalize();
        }
    }
    
    
    public Iterator iterator() {
        this._ptr = -1;
        return this;
    }
    
    public boolean hasNext() {
        return ((this._ptr+1) < size());
    }
    
    public Object next() {
        this._ptr++;
        return null;
    }
    
    public void remove() {
    	throw new UnsupportedOperationException();
    }
    
    public FeatureVector currentExample() {
    	return featureVectors.get(this._ptr);
    }
    
    
    /**
     * 
     * @param eid
     * @param e
     */
    public void addExample(FeatureVector e) {
        featureVectors.add(e);
    }

    /**
	 * return Example isntance
	 * @param index
	 * @return
	 */
    public FeatureVector getExample(int index) {
        return featureVectors.get(index);
    }
    
    public CategoricalFeatureVector getCExample(int index) {
        FeatureVector fv = featureVectors.get(index);
        assert fv instanceof CategoricalFeatureVector;
        return (CategoricalFeatureVector)fv;
    }
    
    private void count() {
        this.numNegative = -1;
        this.numPositive = -1;
        this.numUnlabeled = -1;
        
        int pc = 0;
        int nc = 0;
    	int uc = 0;
        for (int i = 0;i < size();++i) {
            CategoricalFeatureVector cfv = getCExample(i);
            
            if (cfv.isPositive()) {
            	pc++;
            }
            else if (cfv.isNegative()) {
            	nc++;
            }
            
            if (cfv.isUnlabeled()) {
                uc++;
            }
        }
    	
    	this.numNegative = nc;
    	this.numPositive = pc;
        this.numUnlabeled = uc;
    }
    
    public int unlabeledSize() {
    	if (this.numUnlabeled < 0) {
    		count();
    	}
    	return this.numUnlabeled;
    }
    
    public int positiveSize() {
        if (this.numPositive < 0) {
            count();
    	}
    	return this.numPositive;
    }
    
    public int negativeSize() {
    	if (this.numNegative < 0) {
            count();
    	}
    	return this.numNegative;
    }
    
    
    
    
    public int size() {
    	if (size < 0) {
    		this.size = this.featureVectors.size();
    	}
        //return featureVectors.size();
        return this.size;
    }
    
    public void setFeatureDimension(int dim) {
        assert dim > 0;
    	this.featureDimension = dim;
    }
    
    /**
     * return dimension of the feature space
     * @return
     */
    public int featureDimension() {
        assert featureDimension > 0;
        return featureDimension;
    }
    
    public void setIndexor(Indexor indexor) {
    	this.indexor = indexor;
    }
    
    public Indexor getIndex() {
        return indexor;
    }
}
