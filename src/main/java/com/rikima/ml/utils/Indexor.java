/*
 * Created on 2004/09/08
 *
 * Workfile:wordIndexor.java
 * Author:rikitoku
 *
 * Copyright (c) 2002 by Justsystem Corporation. All rights reserved.
 *
 */

package com.rikima.ml.utils;

/**
 * indexor for words
 * @author rikitoku
 * @version $Revision: 1.2 $
 * $Id: WordIndexor.java,v 1.2 2005/09/16 09:40:16 rikitoku Exp $
 */

import java.util.*;
//import gnu.trove.*;

public class Indexor {
	
    // fields -------------
	/** word counter */
    private int count;
    private int maxId;
    
    /** total word length */
    private int totalLength;
    
    /** array for word surface */
    private char[] words;
    
    /** array for word length */
    private int[] wordPointers;
    
    
    /** hashmap instance */
    //private TObjectIntHashMap thash;
    private HashMap<String, Integer> hash;
    
    /** buffer for the total words */
    private List<String> wordsBuffer;
    
    // constructors -------
    
    /**
     * constructor, 
     * generate trie instace and buffer for word surface
     */
    public Indexor() {
        //this.thash = new TObjectIntHashMap();
        this.hash = new HashMap<String, Integer>();
    	this.wordsBuffer = new ArrayList<String>();
    }
    
    // methods ------------
    
    /**
     * add word to indexies, we assumed that id is more that 1
     */
    
    public Integer addEntry(String entry) {
        assert entry != null;
        Object o = hash.get(entry);
        if (!hash.containsKey(entry)) {
            hash.put(entry,new Integer(++count));
            
            wordsBuffer.add(entry);
            totalLength += entry.length();
            return count;
        }
        else {
            return -1 * (Integer)o;
        }
    }

    public Integer addEntry(String entry, int id) {
    	assert entry != null;
        
    	if (!hash.containsKey(entry)) {
            Integer oid = new Integer(id);
    		hash.put(entry, oid);
            
            wordsBuffer.add(entry);
            totalLength += entry.length();
            
            count = hash.size();
            
            return oid;
        }
    	else {
            Integer oid = hash.get(entry);
    		return -1 * oid;
    	}
    }
    
    public int getId(String entry) {
    	if (entry == null || !hash.containsKey(entry)) {
    		return 0;
    	}
        int o = (Integer)hash.get(entry);
        return o;
    }
    
    /**
     * set inner array(wordPointers, wordLengths) and destruct buffer for word
     * @return
     */
    
    public int pack() {
    	
    	if (wordsBuffer == null) {
    		return count;
    	}
    	
        words = new char[totalLength+1];
        wordPointers = new int[2*count+2];
        
        int ptr = 1;
        int c = 2;
        int cc = 0;
        for (int i = 0;i < count;++i) {
            ++cc;
        	String s = (String)wordsBuffer.get(i);//it.next();
            try {
                assert s!=null;
            }
            catch (Error e) {
                e.printStackTrace();
                System.err.println("cc = " + cc);
            }
            System.arraycopy(s.toCharArray(),0,words,ptr,s.length());
            wordPointers[c++] = ptr;
            wordPointers[c++] = s.length();
            ptr += s.length();
        }
        wordsBuffer= null;
        return c;
    }
    
    /**
     * re-generate thew wordBuffer for the incremental add
     *
     */
    public void unpack() {
    	if (wordsBuffer == null) {
            this.wordsBuffer = new ArrayList<String>(count);
    	}
        
        for (int i = 1;i <= count;++i) {
            wordsBuffer.add(getEntry(i));
        }
        words = null;
        wordPointers = null;
    }
    
    /**
     * return word surface associated id
     * @param id word id
     * @return word surface
     */
    public String getEntry(int id) {
        if (id < 1) {
            return null;
        }
        
        if (words == null) {
            if (id <= wordsBuffer.size()) {
                return (String)wordsBuffer.get(id-1);
            }
        	else {
                return null;
        	}
        }
        
        int ptr = 2*(id);
        return new String(words,wordPointers[ptr],wordPointers[ptr+1]);
    }
     
    public String getEntry(Integer id) {
    	return getEntry(id.intValue());
    }
    
    /**
     * return indexed word count 
     * @return
     */
    
    public int count() {
        if (count > 0) {
        	return count;
        }
        else if (maxId > 0) {
        	return maxId;
        }
        else {
            return hash.size();
        }
    }
    
    
    
    /**
     * clear all inner array.
     *
     */
    public void clear() {
        this.count = 0;
        this.totalLength = 0;
        
        this.words = null;
        this.wordPointers = null;
        
        this.wordsBuffer.clear();
    }
}
