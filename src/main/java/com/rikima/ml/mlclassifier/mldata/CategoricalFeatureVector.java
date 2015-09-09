package com.rikima.ml.mlclassifier.mldata;

import java.util.*;

public class CategoricalFeatureVector extends FeatureVector {
    static final boolean DEBUG = true;
    
    // fields ----------------
	public static final int POSITIVE = 1;
    public static final int NEGATIVE = -1;
    
    public static final long SEED = 20100224;
    
    protected int y;
    protected double classValue;
    protected double prob = 0.0;
    
    private boolean unlabeled = false;
    
    private Random rand = new Random(SEED);
    
    // constructors ----------
    public CategoricalFeatureVector(int id, int size) {
        super(id, size);
        classValue = 0.0;
	}
	
	public CategoricalFeatureVector(int id, TreeMap<Integer, Double> buf) {
		super(id, buf);
	}
	
	// methods -----------------
	
    public void setGussianRandomClassValue() {
        this.classValue = (this.rand.nextGaussian() > 0)? POSITIVE:NEGATIVE;
        this.unlabeled = true;
    }
    
    public void setProb(double p ) {
    	this.prob = p;
    }
    
    /*
	public void setClassValue(double cval) {
        assert cval == 1.0 || cval == -1.0;
		if (cval == 1.0) {
            classValue = 1.0;
        }
        else if (cval == -1.0) {
            classValue = -1.0;
        }
    }
	*/

	public void setClassValue(double cval) {
        this.classValue = cval;
    }

	public void setPositiveClass() {
		this.classValue = POSITIVE;
		this.y = POSITIVE;
        this.unlabeled = false;
	}
	
	public void setNegativeClass() {
		this.classValue = NEGATIVE;
        this.y = NEGATIVE;
        this.unlabeled = false;
    }
	
	/*
	public void setClassId(int id) {
		assert id == POSITIVE || id == NEGATIVE;
        if (id == POSITIVE) {
        	classValue = y = POSITIVE;
        }
        else if (id == NEGATIVE) {
            classValue = y = NEGATIVE;
        }
    }
	*/
	
	public int y() {
		return this.y;
	}
	
    public double getClassValue() {
		if (this.isLabeled()) {
			return y();
		}
		else {
            return classValue;
		}
    }
	
	public boolean isPositive() {
		return this.y == POSITIVE && !this.unlabeled;
	}
	
	public boolean isNegative() {
		return this.y == NEGATIVE && !this.unlabeled;
	}
	
	public boolean isLabeled() {
        return (this.y == POSITIVE || this.y == NEGATIVE) && !this.unlabeled;
    }
	
	public boolean isUnlabeled() {
        return this.unlabeled;
    }
    
	public  boolean setLabeled() {
		this.unlabeled = false;
		return this.isLabeled();
	}
	
	public boolean setUnlabeled() {
		this.unlabeled = true;
		return this.isUnlabeled();
	}
	
    public String toString() {
        //String retval = ( (isPositive())? "+1 " : "-1 ") + super.toString();
        String retval = "#" + this.id + " y=" + y + " class value=" + Integer.toString((int)this.classValue) + " unlabeled:" + this.unlabeled;
        retval += " prob=" + this.prob;
        return retval;
	}
}
