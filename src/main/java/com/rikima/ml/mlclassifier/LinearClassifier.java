package com.rikima.ml.mlclassifier;

import java.io.*;
import java.util.*;

import org.json.*;


import com.rikima.ml.mlclassifier.mldata.CategoricalFeatureVector;
import com.rikima.ml.mlclassifier.mldata.FeatureVector;
import com.rikima.ml.mlclassifier.mldata.MLData;
import com.rikima.ml.mlclassifier.mldata.factory.MLDataFactory;
import com.rikima.ml.utils.Indexor;
import com.rikima.ml.utils.Reader;


public class LinearClassifier {
    static boolean DEBUG = false;

    static PrintStream stdout = System.out;
    static PrintStream stderr = System.err;

    static String SPACE = " ";
    static String DELIMITER = ":";


    // fields -------------

    private Indexor indexor;
    private MLData testdata;
    private FeatureVector weightVector;

    private double[] scores;

    ArrayList<HashMap> positiveHitInfos = new ArrayList<HashMap>();
	ArrayList<HashMap> negativeHitInfos = new ArrayList<HashMap>();


    // constructors --------

    /**
     * constructor
     *
     */
    public LinearClassifier(MLData testdata, JSONObject json) throws Exception {
        this.testdata = testdata;
        assert testdata.getIndex() != null;

        this.scores = new double[testdata.size()];

        this.weightVector = constructWeightVector(json);
    }


    // methods -------------

    public FeatureVector getWeightVector() {
    	return this.weightVector;
    }

    /**
     * classify
     *
     * @throws Exception
     */
    public void classify() throws Exception {
    	for (int i = 0;i < testdata.size();++i) {
            CategoricalFeatureVector cfv = testdata.getCExample(i);
            scores[i] = score(cfv);
        }
    }

    public void classifyWithHitInfo() throws Exception {

    	for (int i = 0;i < testdata.size();++i) {
            CategoricalFeatureVector cfv = testdata.getCExample(i);
            scores[i] = score(cfv);

            hitInfo(cfv, positiveHitInfos, negativeHitInfos);
        }
    }

    public void hitInfo(FeatureVector cfv, ArrayList<HashMap> positiveHitInfos, ArrayList<HashMap> negativeHitInfos) {

    	HashMap<String, Double> positiveHits = new HashMap<String, Double>();
    	HashMap<String, Double> negativeHits = new HashMap<String, Double>();

    	for (int i = 0;i < cfv.size();++i) {
            int id = cfv.idByIndex(i);

            if (this.weightVector.hasId(id)) {
                double v_cfv = cfv.valueByIndex(i);
                double v_wv =  this.weightVector.valueById(id);

                double v = v_cfv * v_wv;

                String t = this.indexor.getEntry(id);

                if (v_wv >= 0) {
                	positiveHits.put(t, v);
                }
                else {
                	negativeHits.put(t, v);
                }
            }
        }

    	positiveHitInfos.add(positiveHits);
    	negativeHitInfos.add(negativeHits);
    }

    public double score(FeatureVector fv) {
    	return fv.dot(this.weightVector);
    }

    /**
     * return score to classify
     *
     * @param svmdata
     * @return
     */
    public double score(String svmdata) {
    	double sc = 0;

    	String[] ss = svmdata.split(SPACE);
    	for (String s : ss) {
            String[] ss2 = s.split(DELIMITER);
            if (ss2.length != 2) {
            	continue;
            }

            int fid = this.indexor.getId(ss2[0]);
            double val = Double.parseDouble(ss2[1]);


            if (fid > 0) {
                sc += this.weightVector.valueById(fid) * val;
            }

            try {
                assert !Double.isNaN(sc);
           }
            catch (Error err) {
                err.printStackTrace();
                stderr.println(fid + " " +  val);
            }
    	}
        assert !Double.isNaN(sc);
    	return sc;
    }


    /**
     * print accuracy results
     *
     * @throws IOException
     */
    public void printResults() throws IOException {

    	int cor = 0;
    	int miss = 0;

    	int pp = 0;
    	int pn = 0;
    	int np = 0;
    	int nn = 0;

    	for (int i = 0;i < testdata.size();++i) {
    		CategoricalFeatureVector cfv = testdata.getCExample(i);
            stdout.println(scores[i] + " " + cfv.y() + " " + cfv.toString());

            // 正解
            if (cfv.y() * scores[i] > 0) {
                cor++;

                if (cfv.isPositive()) {
                	pp++;
                }
                else {
                	nn++;
                }
            }
            // 不正解
    		else if (cfv.y() * scores[i] < 0) {
                miss++;

    			if (DEBUG) {
                    stderr.println(cfv.getClassValue() + " " + scores[i]);
    			}

                if (cfv.textInfo != null && cfv.textInfo.length() > 0) {
    				stderr.println("---");
                    stderr.println("#error instance");
    				stderr.println("#score=" + scores[i]);
                    stderr.println(" " + cfv.textInfo);
    			}

    			if (cfv.isPositive()) {
    				pn++;
    			}
    			else {
    				np++;
    			}
    		}
            // score = 0 == positive
    		else {
    			if (cfv.isPositive()) {
    				cor++;
                    pp++;
    			}
    			else {
    				miss++;
    				np++;
    			}
    		}
        }

        stderr.println("------");
        stderr.println("# acc = " + (double)cor / (cor + miss) + " = " + cor + " / (" + cor + " + " + miss + ")" );
        stderr.println("# R = " + (double)pp / (pp +pn ) + " = " + pp + " / (" + pp + " + " + pn + ")" );
        stderr.println("# P = " + (double)pp / (pp + np) + " = " + pp + " / (" + pp + " + " + np + ")" );
        stderr.println("# pp pn np nn : " + pp + " " + pn + " " + np + " " + nn);
    }


    public static Model readModel(String model) throws Exception {
        // load model json
        JSONObject jo = new JSONObject(Reader.read(model));

    	int size = jo.getInt("count");
    	double c = jo.getDouble("c");

    	JSONArray ja = jo.getJSONArray("weights");

    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	HashMap<String, Integer> map = new HashMap<String, Integer>();
    	for (int i = 0; i < size;++i) {

            JSONObject o = (JSONObject)ja.get(i);

            if (DEBUG) {
                stderr.println(o.toString());
            }

            String r =  o.getString("rep");

            if (!map.containsKey(r)) {
            	map.put(r, map.size()+1);
            }


            double v = o.getDouble("val");
            buf.put(map.get(r), v);
        }

    	FeatureVector wv = new FeatureVector(0, buf);

    	Model m = new Model();
    	m.weightVector = wv;
    	m.c = c;
    	return m;
    }

    /**
     *
     * @param model
     * @return
     */
    public static FeatureVector readWeightVector(String model) throws Exception {
        // load model json
        JSONObject jo = new JSONObject(Reader.read(model));

    	int size = jo.getInt("count");
    	JSONArray ja = jo.getJSONArray("weights");

    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	HashMap<String, Integer> map = new HashMap<String, Integer>();
    	for (int i = 0; i < size;++i) {

            JSONObject o = (JSONObject)ja.get(i);

            if (DEBUG) {
                stderr.println(o.toString());
            }

            String r =  o.getString("rep");

            if (!map.containsKey(r)) {
            	map.put(r, map.size()+1);
            }


            double v = o.getDouble("val");
            buf.put(map.get(r), v);
        }

        return new FeatureVector(0, buf);


    }

    /**
     * construct weight vector from model json object
     *
     * @param jo
     * @return
     * @throws JSONException
     */
    FeatureVector constructWeightVector(JSONObject jo) throws JSONException {
        if (this.testdata != null) {
        	this.indexor = this.testdata.getIndex();
        }
        else {
        	this.indexor = new Indexor();
        }

    	int size = jo.getInt("count");
    	JSONArray ja = jo.getJSONArray("weights");

    	TreeMap<Integer, Double> buf = new TreeMap<Integer, Double>();
    	for (int i = 0; i < size;++i) {

            JSONObject o = (JSONObject)ja.get(i);

            if (DEBUG) {
                stderr.println(o.toString());
            }

            int fid = this.indexor.addEntry(o.getString("rep")); // getId(o.getString("rep"));
            if (fid < 0) {
            	fid *= -1;
            }
            double v = o.getDouble("val");
            buf.put(fid, v);
        }

        return new FeatureVector(0, buf);
    }

    // main ----------------

    /**
	 * print accuracy results
	 *
	 * @throws IOException
	 */
	public void printResultsWithHitInfo() throws IOException {

		int cor = 0;
		int miss = 0;

		int pp = 0;
		int pn = 0;
		int np = 0;
		int nn = 0;

		assert testdata.size() == this.positiveHitInfos.size();
		assert testdata.size() == this.negativeHitInfos.size();

		for (int i = 0;i < testdata.size();++i) {
			CategoricalFeatureVector cfv = testdata.getCExample(i);
	        stdout.println(scores[i] + " " + cfv.y() + " " + cfv.toString());

	        stderr.println("---");
	        stderr.println(scores[i] + " " + cfv.y() + " " + cfv.toString());


	        HashMap<String, Double> positiveHits = this.positiveHitInfos.get(i);
	        HashMap<String, Double> negativeHits = this.negativeHitInfos.get(i);

            stderr.println("# positive hits:");
            for (Iterator<String> iter = positiveHits.keySet().iterator();iter.hasNext();) {
	        	String t = iter.next();
	        	double v = positiveHits.get(t);
                stderr.print(String.format("%s:%f ", t, v));
            }
            stderr.println();

            stderr.println("# negative hits:");
            for (Iterator<String> iter = negativeHits.keySet().iterator();iter.hasNext();) {
	        	String t = iter.next();
                double v = negativeHits.get(t);
                stderr.print(String.format("%s:%f ", t, v));
            }
            stderr.println();



	        if (cfv.y() * scores[i] > 0) {
	            cor++;

	            if (cfv.isPositive()) {
	            	pp++;
	            }
	            else {
	            	nn++;
	            }
	        }
			else if (cfv.y() * scores[i] < 0) {
	            miss++;

				if (DEBUG) {
	                stderr.println(cfv.getClassValue() + " " + scores[i]);
				}

				if (cfv.textInfo.length() > 0) {
					stderr.println("---");
	                stderr.println("#error instance");
					stderr.println("#score=" + scores[i]);
	                stderr.println(" " + cfv.textInfo);
				}

				if (cfv.isPositive()) {
					pn++;
				}
				else {
					np++;
				}
			}
	        // score = 0 == positive
			else {
				if (cfv.isPositive()) {
					cor++;
	                pp++;
				}
				else {
					miss++;
					np++;
				}
			}
	    }

	    stderr.println("------");
	    stderr.println("# acc = " + (double)cor / (cor + miss) + " = " + cor + " / (" + cor + " + " + miss + ")" );
	    stderr.println("# R = " + (double)pp / (pp +pn ) + " = " + pp + " / (" + pp + " + " + pn + ")" );
	    stderr.println("# P = " + (double)pp / (pp + np) + " = " + pp + " / (" + pp + " + " + np + ")" );
	    stderr.println("# pp pn np nn : " + pp + " " + pn + " " + np + " " + nn);
	}


	/**
     * main
     */
    public static void main(String[] args) {

    	if (args.length < 4) {
    		stderr.println("please input -m [model json] -i [test svmdata]");
    		System.exit(1);
        }

    	String fname = null;
    	String json = null;
        boolean saveDoc = false;
        boolean l2normalize = false;
        boolean withHitInfo = false;
        for (int i = 0;i < args.length;++i) {
    		if (args[i].equals("-i")) {
                fname = args[++i];
            }
    		else if (args[i].equals("-m")) {
    			json = args[++i];
    		}
    		else if (args[i].equals("-s") || args[i].equals("--save_docs")) {
    			saveDoc = true;
            }
    		else if (args[i].equals("-n2") || args[i].equals("--l2normalize")) {
                l2normalize = true;
            }
    		else if (args[i].equals("-hi") || args[i].equals("--with_hit_infos")) {
                withHitInfo = true;
    		}
        }

    	assert fname != null;
        assert json != null;


        try {
            // load model json
            JSONObject jo = new JSONObject(Reader.read(json));


            // load test data
            MLDataFactory mf = new MLDataFactory(fname);
            MLData testdata = null;
            if (saveDoc) {
                testdata = mf.getWithIndexor(saveDoc);
            }
            else {
                testdata = mf.getWithIndexor();
            }

            if (l2normalize) {
            	testdata.l2normalize();
            }

            LinearClassifier self = new LinearClassifier(testdata, jo);

            if (!withHitInfo) {
                self.classify();
                self.printResults();
            }
            else {
                self.classifyWithHitInfo();
                self.printResultsWithHitInfo();
            }
        }
        catch (Exception e) {
    		e.printStackTrace();
    	}



    }
}
