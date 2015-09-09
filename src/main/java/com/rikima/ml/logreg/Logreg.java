package com.rikima.ml.logreg;

import java.io.*;
import java.util.*;

import com.rikima.ml.mlclassifier.AbstractTrainer;
import com.rikima.ml.mlclassifier.mldata.*;
import com.rikima.ml.mlclassifier.mldata.factory.*;

public class Logreg extends AbstractTrainer implements LossFunction {
    static boolean DEBUG = false;
    static PrintStream stderr = System.err;
    static PrintStream stdout = System.out;

    static boolean verbose = false;

    // fields -------------------

    boolean useWeightCollection = false;

    public static final int L1 = 1;
    public static final int L2 = 2;

    protected MLData mldata;
    protected Minimizable minimizer;
    protected double c;

    protected double positiveWeight = 1.0;

    // constructros --------------
    /**
     * constructor
     *
     */
    public Logreg(MLData mldata, double c, int type) {
        this.mldata = mldata;
        this.minimizer = (type == L1)? (new OWLQNState(this, mldata, c)) : (new LQNState(this, mldata, c));
        this.c = c;

        if (useWeightCollection) {
        	weightCorrection();
        }
    }

	public Logreg(MLData mldata, double c, int type, double positiveWeight) {
		this(mldata, c, type);
		this.positiveWeight = positiveWeight;
	}


	// methods --------------

	/**
	 * weight correction for gic experiments
	 */
	protected void weightCorrection() {
        double p = 0;
		double n = 0;
		for (int i = 0;i < mldata.size();++i) {
			CategoricalFeatureVector cfv = mldata.getCExample(i);

			if (cfv.y() > 0) {
				p += 1.0;
			}
			else {
                n += 1.0;
			}
		}

		double w = n/p;
		w = 1.0;
		for (int i = 0;i < mldata.size();++i) {
			CategoricalFeatureVector cfv = mldata.getCExample(i);

			if (cfv.y() > 0) {
                cfv.setWeight(w);
			}
		}


	}

	public double getC() {
		return c;
	}

	public double likellihood() {
		return this.eval(this.minimizer.w(), this.minimizer.grad());
	}

	/**
	 * eval loss function and gradient
	 *
	 */

    public double eval(double[] input, double[] gradient) {
		if (DEBUG) {
            stderr.println("#Logreg#eval()");
		}

        Arrays.fill(gradient, 0);
        double loss = 1.0;

        for (int i = 0 ; i < mldata.size(); ++i) {
            CategoricalFeatureVector cfv = mldata.getCExample(i);

            if (cfv.isUnlabeled()) {
            	continue;
            }

            double l_i = this.eachLoss(cfv, input, gradient);
            loss += l_i;
        }

        assert loss > 0;
        return loss;
    }

    /**
     *
     *
     * @param cfv
     * @param gradient
     * @return
     */
    private double eachLoss(CategoricalFeatureVector cfv, double[] input, double[] gradient) {
        assert cfv.isLabeled();

        // score = y * w * x
		double score = cfv.y() * cfv.dot(input);

        double insLoss, insProb;
        if (score < -30) {
			insLoss = - score;
            insProb = 0;
        }
        else if (score > 30) {
			insLoss = 0;
			insProb = 1;
		}
        else {
            double temp = 1.0 + Math.exp(-score);
            insLoss = Math.log(temp);
            insProb = 1.0 / temp;
		}

        if (cfv.getWeight() != 1.0) {
        	insLoss *= cfv.getWeight();
        }

        double coef = - cfv.y() * (1.0 - insProb);
        //double coef = (1.0 - insProb);
        for (int j = 0;j < cfv.size();++j) {
            gradient[cfv.idByIndex(j)-1] += cfv.valueByIndex(j) * coef;;
        }

        return insLoss;
    }


    private double scoreToProb(double score) {
        double insProb;
        if (score < -30) {
			//insLoss = -score;
            insProb = 0;
        }
        else if (score > 30) {
			//insLoss = 0;
			insProb = 1;
		}
        else {
            double temp = 1.0 + Math.exp(-score);
            //insLoss = Math.log(temp);
            insProb = 1.0 / temp;
		}

        return insProb;
    }

    /**
     *
     * @param cfv
     * @param wv
     * @return
     */
	public static double prob(CategoricalFeatureVector cfv, FeatureVector wv) {
        // score = - y * w * x
		double score = cfv.y() * cfv.dot(wv);

        double insProb;
        if (score < -30) {
			//insLoss = -score;
            insProb = 0;
        }
        else if (score > 30) {
			//insLoss = 0;
			insProb = 1;
		}
        else {
            double temp = 1.0 + Math.exp(-score);
            //insLoss = Math.log(temp);
            insProb = 1.0 / temp;
		}

        return insProb;
    }


	/**
	 *
	 * @param idx
	 * @param grad
	 */
    public void getGradientElement(int idx, FeatureVector wv, double[] grad) {
    	CategoricalFeatureVector cfv = this.mldata.getCExample(idx);

        // score = y * w * x
		double score = cfv.y() * cfv.dot(wv);
        double insProb = scoreToProb(score);

        double coef = - cfv.y() * (1.0 - insProb);
    	for (int j = 0;j < cfv.size();++j) {
            grad[cfv.idByIndex(j)-1] = cfv.valueByIndex(j) * coef;
        }
    }

	/**
	 * train via minimizer
	 *
	 */
    public FeatureVector train() {
        return this.minimizer.minimize();
    }

    /**
     * return mldata
     */
    public MLData getMLData() {
    	return mldata;
    }

    public void reset(FeatureVector wv) {
    	minimizer.reset(wv);
    }

    public void init(){
        minimizer.init();
    }

    public double[] getGradient() {
    	return this.minimizer.grad();
    }

    // main --------------

	/**
	 * @param args
	 */
	public static void main(String[] args) {

		if (args.length < 4) {
			stdout.println("please input -i [file name] -c [l1 weight] (-o [output model]) -l1 or -l2 -n2? -b [true/false]");
            System.exit(1);
		}

		//Minimizable m = null;
        boolean l2normalize = false;
		int type = L1;

		String fname = null;
		String model = null;
		double c = 1.0;
		double pw = 1.0;

		boolean verbose = true;
		double ur = 0.0;
        for (int i = 0;i < args.length;++i) {

			if (args[i].equals("-i")) {
                fname = args[++i];
			}
			else if (args[i].equals("-c")) {
				c = Double.parseDouble(args[++i]);
			}
			else if (args[i].equals("-m")) {
                model = args[++i];
            }
            else if (args[i].startsWith("-l")) {
                if (args[i].charAt(2) == '1') {
                	type = L1;
                }
                else if (args[i].charAt(2) == '2') {
                    type = L2;
                }

			}
            else if (args[i].startsWith("-n2")) {
                l2normalize = true;
                stderr.println("# L2 normalization use.");
            }
            else if (args[i].startsWith("-pw")) {
                pw = Double.parseDouble(args[++i]);
                stderr.println("# positive weight=" + pw);
            }
            else if (args[i].startsWith("-v") || args[i].startsWith("--verbose")) {
                verbose = true;
            }
            else if (args[i].startsWith("-u") || args[i].startsWith("--unlabled_ratio")) {
                ur = Double.parseDouble(args[++i]);
            }
            else if (args[i].startsWith("-b")) {
                MLDataFactory.useBinaryFeature = Boolean.parseBoolean(args[++i]);
            }
        }

        if (model == null) {
            model = fname + ".model.L" + type;
        }

		try {
            MLDataFactory mf = new MLDataFactory(fname);
            MLData mldata = mf.getWithIndexor();

            if (pw != 1.0) {
            	mldata.setPositiveWeight(pw);
            }

            if (l2normalize) {
                mldata.l2normalize();
            }

            if (ur > 0.0) {
            	mldata.setUnlabeled(ur);
            }

            Logreg.verbose = verbose;
            Logreg self = new Logreg(mldata, c, type);

            FeatureVector wv = self.train();

            self.outputWeightVector(model, wv);
        }
		catch (Exception e) {
			e.printStackTrace();
		}

	}

}
