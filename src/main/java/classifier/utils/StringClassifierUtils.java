package classifier.utils;

import classifier.Classifier;
import classifier.spam.Label;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.DirectoryStream;
import java.nio.file.Files;
import java.nio.file.Path;

public class StringClassifierUtils {

	private static final Logger LOGGER = LoggerFactory.getLogger(StringClassifierUtils.class);

	// private constructor
	private StringClassifierUtils() {}

	/**
	 * Classifies the text of a given file
	 * @param classifier the classifier to categorize the {@link String} instance to a {@link classifier.spam.Label}
	 * @param path the path to the target file
	 * @return the predicted {@link classifier.spam.Label}
	 * @throws IOException
	 */
	public static Label classifyFile(Classifier<Label, String> classifier, java.nio.file.Path path) throws IOException {
		byte[] encoded = Files.readAllBytes(path);
		String text = new String(encoded, StandardCharsets.UTF_8);
		Label label = classifier.classify(text);
		LOGGER.info(label + ", " + path.toAbsolutePath());
		return label;
	}

	/**
	 * Recursively classify all files in a given directory
	 * @param classifier the classifier to categorize the {@link String} instance to a {@link classifier.spam.Label}
	 * @param dir the path to the target directory
	 * @throws IOException
	 */
	public static void classifyDir(Classifier<Label, String> classifier, java.nio.file.Path dir) throws IOException {
		LOGGER.info("classifying dir: " + dir);
		// try-with-resources automatically closes the DirectoryStream upon exit
		try (DirectoryStream<Path> stream = Files.newDirectoryStream(dir)) {
			for (java.nio.file.Path entry : stream) {
				if (Files.isDirectory(entry)) {
					classifyDir(classifier, entry);
				} else {
					classifyFile(classifier, entry);
				}
			}
		}
	}

}
