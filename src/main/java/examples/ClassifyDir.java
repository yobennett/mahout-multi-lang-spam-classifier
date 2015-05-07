package examples;

import classifier.Classifier;
import classifier.spam.NaiveBayesClassifier;
import classifier.utils.StringClassifierUtils;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ClassifyDir {

	public static void main(String... args) {
		try {
			if (args.length <= 0) {
				throw new IllegalArgumentException("Must have a valid path to target directory to classify.");
			}

			// path to target directory
			Path path = Paths.get(args[0]);

			// set up a Naive Bayes classifier
			// in our case we are classifying String objects (i.e. posts)
			// which are classified into labels that are Strings ("spam", "ham")
			Classifier<String,String> classifier = new NaiveBayesClassifier();

			// classify all entries in the target directory
			StringClassifierUtils.classifyDir(classifier, path);

		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
