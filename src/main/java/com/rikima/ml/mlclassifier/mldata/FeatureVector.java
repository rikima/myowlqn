/**
 * 
 */
package com.rikima.ml.mlclassifier.mldata;

/**
 * @author rikitoku
 *
 */

import java.util.*;

public class FeatureVector implements Iterator {
	static final boolean DEBUG = true;
    
	// fields ----------------
    protected int id = -1;
    protected double _l1norm = -1.0;
	protected double _l2norm = -1.0;
	
    
    private double[] values;
	private int[] ids;
	
	private TreeMap<Integer, Double> id2values;
	
	private int ptr = -1;
    private double weight = 1.0;
    
    public String textInfo;
    
    // constructors ----------
	
    protected FeatureVector(int id) {
		this.id = id;
	}
	
    public FeatureVector(int id, int size) {
        this.id = id;
        
    	this.values = new double[size];
        this.ids = new int[size];
        
        for (int i = 0;i < size;++i) {
        	ids[i] = i+1;
        }
    }
    
    /**
     * copy constructor
     * @param v
     */
    
    public FeatureVector(FeatureVector v) {
    	this(v.id(), v.size());
        System.arraycopy(v.ids(), 0, this.ids, 0, v.size());
    	System.arraycopy(v.values(), 0, this.values, 0, v.size());
    }
    
    /**
     * constructor via TreeMap 
     * 
     * @param buf
     */
    public FeatureVector(int id, TreeMap<Integer, Double> buf) {
    	
    	this(id, buf.size());
    	
    	int i = 0;
    	for (Iterator<Map.Entry<Integer, Double>> iter = buf.entrySet().iterator();iter.hasNext(); ) {
    		
            Map.Entry<Integer, Double> me = iter.next();
    		
            this.ids[i] = me.getKey();
            this.values[i] = me.getValue();
            
            i++;
        }
    }
    
    
    // methods --------------
    
    public FeatureVector copyFrom(FeatureVector fv) {
    	assert fv.size() == this.size();
    	
        for (int i = 0;i < fv.size();++i) {
        	assert this.ids[i] == fv.idByIndex(i);
        	
        	this.values[i] = fv.valueByIndex(i);
        }
        return this;
    }
    
    public double l1norm() {
        if (_l1norm > 0) {
        	return _l1norm;
        }
        
        _l1norm = 0;
        for (double v : values) {
            assert v >= 0;
        	_l1norm += v;
        }
        
        try {
            assert _l1norm >= 0;
        }
        catch (Error err) {
            err.printStackTrace();
            _l1norm = 0;
        }
        
        return _l1norm;
    }
    
    public void l1normalize() {
        double l1 = l1norm();
        try {
            assert l1 > 0;
        }
        catch (Error err) {
        	err.printStackTrace();
        	
        }
        
        for (int i = 0;i < size(); ++i) {
            setValueByIndex(i, valueByIndex(i) / l1);
        }
        init();
        assert Math.abs( l1norm() - 1.0 ) < 1.0e-5;
    }
    
    
    public Iterator iterator() {
    	this.ptr = -1;
    	return this;
    }
    
    public boolean hasNext() {
    	return ((this.ptr+1) < size());
    }
    
    public Object next() {
        this.ptr++;
    	return null;
    }
    
    public void remove() {
        throw new UnsupportedOperationException();
    }
    
    public int currentId() {
    	return ids[ptr];
    }
    
    public double currentValue() {
    	return values[ptr];
    }
    
    protected void init() {
    	this._l1norm = -1;
    	this._l2norm = -1;
        
    	this.ptr = -1;
    }
    
    public int id() {
    	return id;
    }
    
    
    public int[] ids() {
    	return ids;
    }
    
    public double[] values() {
    	return values;
    }

    
    /**
	 * set value
	 */
    /*
	public void setValueById(int id, double val) {
        assert checkIndex(id);
		values[id] = val;
	}
    */
    
	public void setValueByIndex(int idx, double val) {
        assert checkIndex(idx);
		values[idx] = val;
		init();
    }
    
    public void appendValueByIndex(int idx, double val) {
        setValueByIndex(idx, valueByIndex(idx) + val);
	}
	
    public FeatureVector plus(FeatureVector fv, double coef) {
        if (this.id2values == null) {
        	this.id2values = new TreeMap<Integer, Double>();
        }
        this.id2values.clear();
        
        for (int i = 0;i < this.size();++i) {
            int id = this.idByIndex(i);
            this.id2values.put(id, this.valueByIndex(i));
    	}
                
    	for (int i = 0;i < fv.size();++i) {
            int id = fv.idByIndex(i);
            double v = 0;
            if (this.id2values.containsKey(id)) {
                v = this.id2values.get(id);
            }
            
            this.id2values.put(id, v + fv.valueByIndex(i) * coef);
            
    	}
       
        return new FeatureVector(this.id, this.id2values);
	}
	
	public void plus(FeatureVector fv) {
       plus(fv, 1.0);
    }
	
	public void minus(FeatureVector fv) {
		 plus(fv, -1.0);
	}
	
	
	/**
	 * get 
	 * @param index
	 * @return
	 */
    public double valueByIndex(int idx) {
        assert checkIndex(idx);
        
        return values[idx];
    }

    public double valueById(int id) {
    	assert id > 0;
    	
        int index = Arrays.binarySearch(ids, id);
    	if (index < 0) {
    		return Double.NaN;
    	}
        return values[index];
    }
    
    
    private boolean checkIndex(int idx) {
        if (idx < 0) {
            return false;
        }
        return true;
    }
    
    /**
     * 
     * @return
     */
    public int checkIndices() {
        int pid = 0;
    	for (int i = 0;i < ids.length;++i) {
            if (pid > ids[i]) {
                return -i;
            }
            pid = ids[i];
    	}
        return ids.length;
    }
        
    public double dot(double[] w) {
    	double retval = 0;
    	for (int i = 0;i < size();++i) {
            retval += w[ idByIndex(i) - 1] * valueByIndex(i);
    	}
    	
        return retval;
    }
    
    
    /**
	 * 
	 * @param fv
	 * @return
	 */
    public double dot(FeatureVector fv) {
        FeatureVector afv = (FeatureVector)fv;
        double retval = 0;
        
        if (this.size() == 0 && fv.size() == 0) {
            return 1.0;
        }
        
        int l1 = this.size();
        int l2 = fv.size();
        for (int i1 = 0, i2 = 0; i1 < l1 && i2 < l2;) {
            int id1 = ids[i1];
            int id2 = fv.idByIndex(i2);
            
            if (id1 == id2) {
                retval += values[i1] * afv.valueByIndex(i2);
                ++i1;
                ++i2;
            }
            else if (id1 > id2) {
                ++i2;
            }
            else {
                ++i1;
            }
        }
        
        return retval;
	}
	
    public FeatureVector times(double coef) {
		assert coef != 0;
        
		if (values != null) {
            for (int i = 0;i < size();++i) {
                values[i] *= coef;
            }
        }
		else if (id2values != null){
			for (Iterator iter = this.id2values.entrySet().iterator();iter.hasNext();) {
				Map.Entry<Integer, Double> e = (Map.Entry<Integer, Double>)iter.next();
                this.id2values.put(e.getKey(), e.getValue()*coef);
			}
		}
		
        init();
        return this;
    }
	
    public boolean hasId(int id) {
        return (Arrays.binarySearch(ids, id) >= 0);
    }
    
    public int size() {
    	assert ids.length == values.length;
    	assert ids.length > 0;
        return ids.length;
	}
	
    /** okasii**/
	public int hashCode() {
        int h = 1;
        for (int i = 0;i < ids.length;++i) {
            int id = ids[i];
            h += h * 137 + id;
        }
		
        h %= 1987;
        return h;
	}
	
	public boolean equals(Object o) {
        if (!(o instanceof FeatureVector)) {
        	return false;
        }

        if (this.compareTo(o) == 0) {
        	return true;
        }
        else {
        	return false;
        }
    }
	
	public int compareTo(Object o) {
        if (!(o instanceof FeatureVector)) {
        	return 1;
        }
		
        FeatureVector fv = (FeatureVector)o;
        
		if (size() > fv.size()) {
			return 1;
		}
		else if (this.size() < fv.size()) {
			return -1;
		}
		
        for (int i = 0;i < ids.length;++i) {
            if (!fv.hasId(ids[i])) {
                return 1;
        	}
        }
        return 0;
    }
	
	public int idByIndex(int index) {
        return ids[index];
    }
    
	public void setIdByIndex(int idx, int fid) {
    	assert checkIndex(idx);
    	ids[idx] = fid;
    }
    
	public void clear() {
        this._l1norm = -1;
        this._l2norm = -1;
		Arrays.fill(values, 0);
	}
    
	public void l2normalize() {
		double n = l2norm();
        times(1.0 / n);
        init();
        double EPS = 1.0E-5;
        if (DEBUG && n > 0) {
            assert Math.abs(l2norm() - 1.0) < EPS;
		}
    }
    
	
	/**
	 * to string 
	 * via svm^light format
	 * 
	 */
	public String toString() {
		StringBuffer buf = new StringBuffer();
		
        for (Iterator iter = this.iterator();iter.hasNext(); ) {
        	iter.next();
        	
        	int id = currentId();
            double val = currentValue();
        	
        	buf.append(id + ":" + val + " ");
        }
        return buf.toString().trim();
    }
    
	public String toString(int limit) {

		StringBuffer buf = new StringBuffer();
        for (Iterator iter = this.iterator();iter.hasNext(); ) {
        	iter.next();
        	
        	int id = currentId();
            double val = currentValue();
        	
        	buf.append(id + ":" + val + " ");
            
            if (limit-- <= 0) {
        		break;
            }
        }
        return buf.toString().trim();
    }
    
	
	
	public double l2norm() {
		if (_l2norm > 0) {
			return _l2norm;
		}
		_l2norm = 0;
		
		for (double v : values) {
			_l2norm += v * v;
		}
        
		_l2norm = Math.sqrt(_l2norm);
        return _l2norm;
	}
	
	public void setWeight(double w) {
		this.weight = w;
	}
	
	public double getWeight() {
		return this.weight;
	}
}