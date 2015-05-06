package classifier.spam;

import classifier.analyses.Analysis;
import classifier.analyses.EnglishAnalysis;
import org.apache.mahout.math.Vector;

import java.io.IOException;

public class EnglishNaiveBayesClassifier extends NaiveBayesClassifier {

	public EnglishNaiveBayesClassifier(String propertiesName) {
		super(propertiesName);
	}

	/**
	 * Returns analysis of instance
	 * @param instance text to analyze
	 * @return analysis of instance
	 * @throws IOException
	 */
	public Analysis analysis(String instance) throws IOException {
		return new EnglishAnalysis(instance, dictionary, frequencies);
	}

	/**
	 * Classifies instance
	 * @param instance text to analyze
	 * @return classification of instance to {@link classifier.spam.Label}
	 * @throws java.io.IOException
	 */
	@Override
	public Label classify(String instance) throws IOException {
		Analysis analysis = analysis(instance);
		Vector instanceVector = analysis.instanceVector();

		Vector probabilitiesVector = classifier.classifyFull(instanceVector);
		double bestScore = -Double.MAX_VALUE;
		int bestLabelId = -1;
		for (Vector.Element e : probabilitiesVector.all()) {
			int labelId = e.index();
			double score = e.get();
			if (bestScore < score) {
				bestScore = score;
				bestLabelId = labelId;
			}
		}
		return Label.valueOf(labels.get(bestLabelId));
	}

}
