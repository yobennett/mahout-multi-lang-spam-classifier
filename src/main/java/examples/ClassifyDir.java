package examples;

import classifier.spam.NaiveBayesClassifier;

import java.io.IOException;
import java.nio.file.Paths;
import java.util.Arrays;

public class ClassifyDir {

	public static void main(String... args) {
		try {
			NaiveBayesClassifier sc = new NaiveBayesClassifier();
			sc.classifyDir(Paths.get(args[0], Arrays.copyOfRange(args, 1, args.length)));
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

}
