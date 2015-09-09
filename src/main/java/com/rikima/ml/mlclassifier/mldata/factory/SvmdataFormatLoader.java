package com.rikima.ml.mlclassifier.mldata.factory;


import java.io.*;
import java.util.*;


import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.utils.Indexor;

public class SvmdataFormatLoader implements Loader {
    static final boolean DEBUG = false;

    // fields ----------------------------
    static final String COMMENT = "#";
    static final String DELIMITER = ":";

    protected boolean useZeroId = true;

    protected boolean isWeighted = false;
    protected boolean indexFeature = true;

    private int cnt;

    protected Indexor featureIndexor;
    protected TreeMap<Integer, Double> featureBuf;

    protected String file;
    protected MLData mldata;

    protected ArrayList<String> docs;
    // constructors ----------------------


    // constructors ----------------------

    public SvmdataFormatLoader(String file) {
    	this.file = file;
        this.mldata = new MLData();
        this.featureIndexor = new Indexor();

        if (useZeroId) {
        	this.featureIndexor.addEntry("0");
        }

        this.featureBuf = new TreeMap<Integer, Double>();
        this.docs = new ArrayList<String>();


    }

    // methods ------------------------------
    /**
     * return mldata
     */
	public Object get() throws Exception{
        load();
        mldata.setFeatureDimension(featureIndexor.count());
        return mldata;
	}

	public Object getWithIndexor() throws Exception {
		return getWithIndexor(false);
    }

	public Object getWithIndexor(boolean saveDoc) throws Exception {
        load(saveDoc);
        mldata.setFeatureDimension(featureIndexor.count());
        mldata.setIndexor(this.featureIndexor);

        return mldata;

	}

	public void load() throws IOException {
        load(false);
	}

    /**
     * load
     */
    public void load(boolean saveDoc) throws IOException {
        System.err.println("reading ... " + file);
        long t = System.currentTimeMillis();
		BufferedReader reader
            = new BufferedReader(new InputStreamReader(new FileInputStream(file),System.getProperty("file.encoding")));

        String line = null;
        while ((line = reader.readLine()) != null) {
            line = line.trim();
            String oline = line;

            int p = line.indexOf(COMMENT+" ");
            if (p >= 0) {
                line = line.substring(0, p);
            }

            p = line.indexOf(DELIMITER);
            if (p < 0) {
                continue;
            }

            if (line.length() == 0) {
            	continue;
            }

            try {
                FeatureVector fv = createCategoricalFeatureVector(line);
                if (saveDoc) {
                    fv.textInfo = oline;
                }

                mldata.addExample(fv);
                ++cnt;
            }
            catch (Exception e) {
                e.printStackTrace();
            }
        }
        reader.close();
        t = System.currentTimeMillis() - t;
        System.err.println(" " + cnt + " done (" + t + " [ms])");
    }

    /**
     * parse and create feature vector instance
     *
     * @param line
     * @return
     * @throws Exception
     */
    private CategoricalFeatureVector createCategoricalFeatureVector(String line) throws Exception {
        CategoricalFeatureVector fv = (CategoricalFeatureVector)createFeatureVector(line);

        try {
        	line = line.trim();
            int p = line.indexOf(' ');
            assert p > 0;

            int pp = (line.charAt(0) == '+')? 1:0;
            String c = line.substring(pp, p);

            double classVal = Double.parseDouble(line.substring(pp, p));

            assert classVal == 1.0 || classVal == 0.0 || classVal == -1.0;

            fv.setClassValue(classVal);
            if (classVal == 1.0 || classVal == -1.0) {
                if (classVal == 1.0) {
            		fv.setPositiveClass();
            	}
            	else if (classVal == -1.0) {
                    fv.setNegativeClass();
            	}
            }
        }
        catch (Exception e) {
        	e.printStackTrace();
        }
        return fv;
    }


    /**
	 * create Example instance
	 * @param line
	 * @return
	 * @throws Exception
	 */
	private FeatureVector createFeatureVector(String line) throws Exception {
		featureBuf.clear();

		line = line.trim();

		// #コメントを削除
		{
            int p = line.indexOf(" #");
            if (p > 0) {
            	line = line.substring(0, p);
            }
		}

		if (useZeroId) {
			featureBuf.put(1,1.0);
		}

        /*
		StringTokenizer st = new StringTokenizer(line, " ");
        String l = st.nextToken();
        if (l.indexOf(DELIMITER) > 0) {
            throw new Exception("parse exception:" + line);
        }
        */

		String orgline = line;

		// yの設定
        {
        	int p = line.indexOf(" ");
            String ystr = line.substring(0, p);

            int y = 0;
            if (ystr.equals("+1")) {
                y = 1;
            }
            else {
            	y = Integer.parseInt(ystr);
            }

            line = line.substring(p).trim();
        }




		String[] ss = line.split(" \\(");
		for (String tk: ss) {
            int p = tk.lastIndexOf(DELIMITER);

            try {
                assert p > 0;
            }
            catch (Error err) {
            	err.printStackTrace();

                System.err.println("feature parse error:" + tk);
                System.err.println("org line:" + orgline);
                continue;
            }


            String idstr = tk.substring(0,p);
            String valstr = tk.substring(p+1);

            int fid = 0;
            if (indexFeature) {
                fid = featureIndexor.addEntry(idstr);
                if (DEBUG) {
                    System.err.println("input idstr=" + idstr
                        + " feature id=" + fid);
                }
            }
            else {
                fid = Integer.parseInt(idstr);
                int ret = featureIndexor.addEntry(idstr,fid);
                if (DEBUG) {
                    System.err.println("input id=" + fid + " ret id=" + ret);
                }
            }

            double val = 0;
            try {
                if (MLDataFactory.useBinaryFeature) {
                    val = 1.0;
                }
                else {
                    val = Double.parseDouble(valstr);
                }
            }
            catch (Exception ex) {
                ex.printStackTrace();
                System.err.println("line:" + line);
                System.err.println("tk:" + tk);
                System.err.println("valstr:" + valstr);
            	System.exit(1);
            }

            if (fid < 0) {
            	fid *= -1;
            }

            assert fid > 0;


            featureBuf.put(fid, val);

        }

        int eid = cnt + 1;

        return new CategoricalFeatureVector(eid, featureBuf);
    }
}
