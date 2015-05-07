package classifier.utils;

import classifier.Classifier;
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

	public static String classifyFile(Classifier<String,String> classifier, java.nio.file.Path path) throws IOException {
		byte[] encoded = Files.readAllBytes(path);
		String text = new String(encoded, StandardCharsets.UTF_8);
		String label = classifier.classify(text);
		LOGGER.info(label + ", " + path.toAbsolutePath());
		return label;
	}

	public static void classifyDir(Classifier<String,String> classifier, java.nio.file.Path dir) throws IOException {
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
