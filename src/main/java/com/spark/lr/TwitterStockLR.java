/*
 * Reference : https://spark.apache.org/docs/1.1.0/
 */
/**
 * @author Dileep Ranganathan
 */
package com.spark.lr;

import scala.Tuple2;

import org.apache.spark.api.java.*;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vectors;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.apache.spark.mllib.regression.LassoModel;
import org.apache.spark.mllib.regression.LassoWithSGD;
import org.apache.spark.mllib.regression.LinearRegressionModel;
import org.apache.spark.mllib.regression.LinearRegressionWithSGD;
import org.apache.spark.mllib.regression.RidgeRegressionModel;
import org.apache.spark.mllib.regression.RidgeRegressionWithSGD;
import org.apache.spark.SparkConf;

public class TwitterStockLR {
	static double stepsize = 1.0;
	public static void main(String[] args) {
		stepsize = Double.parseDouble(args[1]);
		SparkConf conf = new SparkConf().setAppName("Linear Regression");
		JavaSparkContext sc = new JavaSparkContext(conf);

		// Load and parse the data
		String path = "/data/datafile.text";
		JavaRDD<String> data = sc.textFile(path);
		JavaRDD<LabeledPoint> parsedData = data
				.map(new Function<String, LabeledPoint>() {
					public LabeledPoint call(String line) {
						String[] parts = line.split(",");
						String[] features = parts[1].split(" ");
						double[] v = new double[features.length];
						for (int i = 0; i < features.length; i++)
							v[i] = Double.parseDouble(features[i]);
						return new LabeledPoint(Double.parseDouble(parts[0]),
								Vectors.dense(v));
					}
				});

		// Split initial RDD into two... [60% training data, 40% testing data].
		JavaRDD<LabeledPoint> training = parsedData;
		training.cache();
		JavaRDD<LabeledPoint> test = parsedData;

		// Building the model
		int numIterations = Integer.parseInt(args[0]);
		Double mse1 = linearRegression(training, numIterations, test);
		Double mse2 = ridgeRegression(training, numIterations, test);
		Double mse3 = lassoRegression(training, numIterations, test);
		System.out.println("*****************************************************************\n"
				+ "Number of Iterations : " + numIterations + "\n" 
				+ "LinearRegression Mean Squared Error = " + mse1 + "\n"
				+ "RidgeRegression  Mean Squared Error = " + mse2 + "\n"
				+ "LassoRegression  Mean Squared Error = " + mse3 + "\n"
				+ "*****************************************************************\n");
	}

	static Double linearRegression(JavaRDD training, int numIterations,
			JavaRDD test) {
		final LinearRegressionModel model = LinearRegressionWithSGD.train(
				JavaRDD.toRDD(training), numIterations, stepsize);
		

		// Evaluate model on training examples and compute training error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test
				.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<Double, Double>(prediction, point
								.label());
					}
				});
		Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				new Function<Tuple2<Double, Double>, Object>() {
					public Object call(Tuple2<Double, Double> pair) {
						return Math.pow(pair._1() - pair._2(), 2.0);
					}
				}).rdd()).mean();
		// System.out.println("training Mean Squared Error = " + MSE);
		model.toString();
		return MSE;
	}

	static Double ridgeRegression(JavaRDD training, int numIterations,
			JavaRDD test) {
		final RidgeRegressionModel model = RidgeRegressionWithSGD.train(
				JavaRDD.toRDD(training), numIterations, stepsize, 0.01);

		// Evaluate model on training examples and compute training error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test
				.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<Double, Double>(prediction, point
								.label());
					}
				});
		Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				new Function<Tuple2<Double, Double>, Object>() {
					public Object call(Tuple2<Double, Double> pair) {
						return Math.pow(pair._1() - pair._2(), 2.0);
					}
				}).rdd()).mean();
		// System.out.println("training Mean Squared Error = " + MSE);
		return MSE;
	}

	static Double lassoRegression(JavaRDD training, int numIterations,
			JavaRDD test) {
		final LassoModel model = LassoWithSGD.train(JavaRDD.toRDD(training),
				numIterations, stepsize, 0.001);

		// Evaluate model on training examples and compute training error
		JavaRDD<Tuple2<Double, Double>> valuesAndPreds = test
				.map(new Function<LabeledPoint, Tuple2<Double, Double>>() {
					public Tuple2<Double, Double> call(LabeledPoint point) {
						double prediction = model.predict(point.features());
						return new Tuple2<Double, Double>(prediction, point
								.label());
					}
				});
		Double MSE = new JavaDoubleRDD(valuesAndPreds.map(
				new Function<Tuple2<Double, Double>, Object>() {
					public Object call(Tuple2<Double, Double> pair) {
						return Math.pow(pair._1() - pair._2(), 2.0);
					}
				}).rdd()).mean();
		// System.out.println("training Mean Squared Error = " + MSE);
		return MSE;
	}
}