package com.rikima.ml.utils;


import java.io.*;
import java.util.*;

public class ArrayUtils {
    public static double EPS = 1.0E-10;
	
    // fields ----------------------
    private static TreeMap<Double, Integer> value2indices;
	
    // static methods --------------
    
	public static int[] randomRange(int end) {
        return randomRange(end,System.currentTimeMillis());
	}
	
	public static int[] randomRange(int end, long seed) {
		int[] ary = range(end);
        return shuffle(ary,seed);
    }
	
	/**
	 * 
	 * shuffle
	 * @param ary
	 * @return
	 */
	public static int[] shuffle(int[] ary) {
        return shuffle(0,ary,System.currentTimeMillis());
    }
	
	
	/**
	 * 
	 * shuffle
     *
	 * @param start start iundex, which is included. 
	 * @param ary
	 * @return
	 */
    public static int[] shuffle(int start, int[] ary) {
        return shuffle(start,ary,System.currentTimeMillis());
    }
	
	/**
	 * random shuffle inspired by libsvm code
	 * @param ary
	 * @param seed
	 * @return
	 */
	public static int[] shuffle(int[] ary, long seed) {
        return shuffle(0,ary,seed);
    }
	
	/**
	 * random shuffle inspired by libsvm code
	 * 
	 * return new int[] instance.
	 * 
	 * @param start start index, it is processed.
	 * @param ary
	 * @param seed
	 * @return
	 */
	public static int[] shuffle(int start, int[] ary, long seed) {
        assert start >= 0 && start < ary.length;
		int len = ary.length;
        // random shuffle
        Random rand = new Random(seed);
        
        int[] retary = new int[len];
        System.arraycopy(ary,0,retary,0,len);
        for(int i=start;i < len;i++) {
            int j = i + (int)(rand.nextDouble() * (len - i));
            int _ = retary[i]; 
            retary[i] = retary[j]; 
            retary[j] = _;
        }
        return retary;
    }
	
	
	
	public static int[] range(int end) {
        assert end > 0;
		int[] ary = new int[end];
		for (int i = 0;i < end;++i) {
			ary[i] = i;
		}
		return ary;
	}
    
    public static int[] range(int start,int end) {
        int l = end - start;
    	assert l > 0;
        int[] ary = new int[l];
        for (int i = 0;i < l;++i) {
    		ary[i] = i+start;
    	}
    	return ary;
    }
	
	
    public static void copy(double[] src, double[] dest) {
        assert lengthCheck(src,dest);
        System.arraycopy(src,0,dest,0,src.length);
    }
	
    public static void incrementCopy(double[] src, double[] dest) {
    	assert lengthCheck(src,dest);
        for (int i = 0;i < src.length;++i) {
    		dest[i] += src[i];
    	}
    }
    
    public static void incrementCopy(int[] src, int[] dest) {
    	assert src.length == dest.length;
        for (int i = 0;i < src.length;++i) {
    		dest[i] += src[i];
    	}
    }
    
    public static void timesCopy(double[] src, double[] dest) {
    	assert lengthCheck(src,dest);
        for (int i = 0;i < src.length;++i) {
    		dest[i] *= src[i];
    	}
    }
    
    
    public static void times(double[] src, double[] dist, double c) {
        assert src.length == dist.length;
        for (int i = 0;i < src.length;++i) {
        	dist[i] = c * src[i];
        }
    }
    
    
    public static void times(double[] array, double c) {
        assert c != 0;
    	for (int i = 0;i < array.length;++i) {
        	array[i] *= c;
        }
    }
    
    public static void plus(double[] array, double a) {
        for (int i = 0;i < array.length;++i) {
        	array[i] += a;
        }
    }
    
    public static void subtract(double[] array, double a) {
    	for (int i = 0;i < array.length;++i) {
            array[i] -= a;
        }
    }
    
    public static double sum(double[] array) {
        return sum(array, array.length);
    }
    
    public static double sum(double[] array, int len) {
    	assert len > 0;
    	assert len <= array.length;
    	double s = 0;
        for (int i = 0;i < len;++i) {
        	s += array[i];
        }
        return s;
    }
    
    public static double sum(int[] array) {
        double s = 0;
        for (int i = 0;i < array.length;++i) {
        	s += array[i];
        }
        return s;
    }
    
    
    public static void normalize(double[] array) {
        double s = 0;
    	double as = 0;
        for (int i = 0;i < array.length;++i) {
            s += array[i];
            as += Math.abs(array[i]);
        }
        
        if (as > 0) {
            for (int i = 0;i < array.length;++i) {
                array[i] /= s;
            }
        }
    }
    
    private static final boolean lengthCheck(double[] src, double[] dest) {
        return src.length == dest.length;
    }
    
    public static void write(double[] array, PrintStream writer) throws IOException {
        int l = array.length - 1;
        for (int i = 0;i < l;++i) {
            writer.print(Double.toString(array[i]) + ",");
        }
        writer.println();
    }
    
    
    public static void write(double[] array) throws IOException {
        write(array, System.out);
    }
        
    
    public static void print(String label, double[] array) {
        print(label, array, array.length);
    }
    
    public static void print(String label, double[] array, int limit) {
        System.err.print(label + "={");
        int l = limit - 1;
        for (int i = 0;i < l;++i) {
        	System.err.print(array[i] + ",");
        }
        
        System.err.print(array[l]);
        
        if (limit < array.length) {
            System.err.println("...}");
        }
        else {
            System.err.println(" }");
        }
    }
    
    
    
    public static void print(String label, int[] array) {
        System.err.print(label + "={");
    	int l = array.length - 1;
        for (int i = 0;i < l;++i) {
        	System.err.print(array[i] + ",");
        }
        System.err.println(array[l] + "}");
    }
 
    public static void stdoutPrintln(double[] array) {
        println(System.out,array);
    }
    
    public static void println(PrintStream p, String label, double[] array) {
        p.print(label + "={");
        int l = array.length - 1;
        for (int i = 0;i < l;++i) {
            p.print(array[i] + ",");
        }
        p.println(array[l] + "}");
    }
    
    public static void println(PrintStream p, double[] array) {
        p.print("{");
        int l = array.length - 1;
        for (int i = 0;i < l;++i) {
            p.print(array[i] + ",");
        }
        p.println(array[l] + "}");
    }
    
    public static String toString(double[] array) {
        StringBuffer sb = new StringBuffer("{");
        for (int i = 0;i < array.length;) {
            sb.append(array[i++]);
            if (i < array.length) {
                sb.append(",");
            }
        }
        sb.append("}");
        return sb.toString();
    }
    
    /**
     * return true if all array element is less than var.
     * @param array
     * @param var
     * @return
     */
    public static boolean allLess(double[] array, double var) {
        for (int i = 0;i < array.length;++i) {
            if (array[i] >= var) {
                return false;
            }
    	}
    	return true;
    }
    
    public static double[] scale(double[] array,double min) {
    	if (allLess(array,min)) {
    		times(array,1/min);
    	}
    	return array;
    }
    
    
    /**
     * return true if all array element is more than var;
     * @param array
     * @param var
     * @return
     */
    public static boolean allMore(double[] array, double var) {
    	for (int i = 0;i < array.length;++i) {
    		if (array[i] <= var) {
    			return false;
    		}
        }
        return true;
    }
 
    public static double dot(double[] aArray, double[] bArray) {
        assert aArray.length == bArray.length;
        double ret = 0;
        for (int i = 0;i < aArray.length;++i) {
            ret += aArray[i] * bArray[i];
    	}
        return ret;
    }
    
    public static double innerProduct(double[] aArray, double[] bArray) {
        return dot(aArray, bArray);
    }
 
    public static double innerProduct(int[] aArray, int[] bArray) {
        assert aArray.length == bArray.length;
        double ret = 0;
        for (int i = 0;i < aArray.length;++i) {
            ret += aArray[i] * bArray[i];
    	}
        return ret;
    }
    
    public static double l2Diff(double[] aArray, double[] bArray) {
    	assert aArray.length == bArray.length;
        double retVal = 0;
        for (int i = 0;i < aArray.length;++i) {
            
            // ASSERT
            assert !Double.isNaN(aArray[i]);
            assert !Double.isNaN(bArray[i]);
    		
            double d = (aArray[i] - bArray[i]);
            retVal += d * d;
    	}
    	return retVal;
    }
    
    public static double l2norm(double[] aArray) {
    	return innerProduct(aArray, aArray);
    }
    
    public static int minIndex(double[] array) {
        int idx = 0;
    	double min = array[0];
        for (int i = 1;i < array.length;++i) {
    		if (min > array[i]) {
                min = array[i];
                idx = i;
            }
    	}
        
    	try {
            assert idx >= 0;
    	}
    	catch (Error e) {
    		e.printStackTrace();
            print("array=", array);
            System.exit(1);
    	}
        return idx;
    }
    
    /**
     * return index having max value
     * @param aArray
     * @return
     */
    public static int maxIndex(double[] aArray) {
    	double max = -Double.MAX_VALUE;
    	int idx = -1;
    	
    	for (int i = 0;i < aArray.length;++i) {
    		if (aArray[i] > max) {
    			max = aArray[i];
    			idx = i;
    		}
    	}
        
    	try {
            assert idx >= 0;
    	}
    	catch (Error e) {
    		e.printStackTrace();
    		print("aArray=", aArray);
            System.exit(1);
    	}
        return idx;
    }
    
    public static Integer[] maxIndices(double[] aArray, Integer[] indices) {
        if (value2indices == null) {
            value2indices = new TreeMap<Double, Integer>();
    	}
    	value2indices.clear();
        for (int i = 0;i < aArray.length;++i) {
            if (value2indices.get(aArray[i]) != null) {
                aArray[i] += EPS;
            }
        	value2indices.put(-aArray[i], i);
        }
        
        value2indices.values().toArray(indices);
        return indices;
    }
    
    public static double[] randomDoubleArray(int size) {
    	Random rand = new Random(System.currentTimeMillis());
    	return randomDoubleArray(size, rand);
    }
    
    public static double[] randomDoubleArray(int size, Random rand) {
    	double[] ary = new double[size];
    	for (int i = 0;i < size;++i) {
    		ary[i] = rand.nextDouble();
        }
    	return ary;
    }
    
    public static double min(double[] arry) {
    	double min = Double.MAX_VALUE;
    	for (double a : arry) {
            if (a < min) {
            	min = a;
            }
        }
    	return min;
    }
    
    public static double max(double[] arry) {
        double max = -Double.MAX_VALUE;
    	for (double a : arry) {
            if (a > max) {
            	max = a;
            }
        }
    	return max;
    }
    
    public static void log(double[] array) {
        for (int i = 0;i < array.length;++i) {
    		array[i] = Math.log(array[i]);
        }
    }
    
    public static void exp(double[] array) {
    	for (int i = 0;i < array.length;++i) {
            array[i] = Math.exp(array[i]);
        }
    }
}
