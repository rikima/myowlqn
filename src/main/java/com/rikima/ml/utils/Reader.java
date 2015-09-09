package com.rikima.ml.utils;

import java.io.*;



public class Reader {
    static boolean DEBUG = false;
    
	static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;
	
    
	
	/**
	 * read 
	 * 
     * @param fname
	 * @return
	 * @throws IOException
	 */
	public static String read(String fname) throws IOException {
        
		stderr.print("reading " + fname + " ...");
		BufferedReader br = new BufferedReader( new InputStreamReader( new FileInputStream(fname), System.getProperty("file.encoding")));
        StringBuffer sb = new StringBuffer();
        
        String l = null;
        
        while ((l = br.readLine()) != null) {
            if (DEBUG) {
            	stderr.println(l);
            }
        	sb.append(l);
        }
        
		br.close();
        stderr.println(" .done");
        
        return sb.toString();
	}
	
	
	
}
