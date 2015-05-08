package examples;

import classifier.Classifier;
import classifier.spam.Label;
import classifier.spam.NaiveBayesClassifier;
import classifier.utils.StringClassifierUtils;
import com.google.common.base.Joiner;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

public class ClassifyDir {

	private static final Logger LOGGER = LoggerFactory.getLogger(ClassifyDir.class);

	public static void main(String... args) {
		try {
			if (args.length <= 0) {
				throw new IllegalArgumentException("Must have a valid path to target directory to classify.");
			}

			// path to target directory
			Path path = Paths.get(args[0]);

			LOGGER.info("classifying into: " + Joiner.on(", ").join(Label.values()));

			// set up a Naive Bayes classifier
			// in our case we are classifying String objects (i.e. posts)
			// which are classified into a label (spam, ham, unsure)
			Classifier<Label, String> classifier = new NaiveBayesClassifier("classifier.properties");

			// classify all entries in the target directory and its subdirectories
			StringClassifierUtils.classifyDir(classifier, path);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
