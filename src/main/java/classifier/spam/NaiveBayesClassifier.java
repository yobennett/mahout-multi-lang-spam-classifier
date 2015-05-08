package classifier.spam;

import classifier.Classifier;
import classifier.analyses.Analysis;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.mahout.classifier.naivebayes.BayesUtils;
import org.apache.mahout.classifier.naivebayes.NaiveBayesModel;
import org.apache.mahout.classifier.naivebayes.StandardNaiveBayesClassifier;
import org.apache.mahout.common.Pair;
import org.apache.mahout.common.iterator.sequencefile.SequenceFileIterable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

public class NaiveBayesClassifier implements Classifier<Label, String> {

	private static final Logger LOGGER = LoggerFactory.getLogger(NaiveBayesClassifier.class);

    private final Configuration conf;
    private final Properties prop;
    private final NaiveBayesModel model;
    protected final StandardNaiveBayesClassifier classifier;
    protected final Map<Integer, String> labels;
    protected final Map<String, Integer> dictionary;
    protected final Map<Integer, Long> frequencies;

    public NaiveBayesClassifier(String propertiesName) {
        this.conf = new Configuration();

        try {
            this.prop = loadProperties(propertiesName);
            this.model = NaiveBayesModel.materialize(new Path(prop.getProperty("classifier.model.dir")), conf);
            this.classifier = new StandardNaiveBayesClassifier(model);
            this.labels = readLabelIndex(new Path(prop.getProperty("classifier.model.labelindex")));
            this.dictionary = readDictionary(new Path(prop.getProperty("classifier.model.dictionary.path")));
            this.frequencies = readFrequencies(new Path(prop.getProperty("classifier.model.frequencies.path")));
        } catch (IOException e) {
            throw new IllegalStateException("Missing or invalid classifier properties at " + propertiesName + ".");
        }
    }

	/**
	 * Loads properties
	 * @param name properties filename
	 * @return properties
	 * @throws IOException
	 */
    private Properties loadProperties(String name) throws IOException {
        Properties properties = new Properties();
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        try (InputStream stream = loader.getResourceAsStream(name)) {
            properties.load(stream);
	        LOGGER.info("loaded properties from " + name);
            return properties;
        }
    }

	/**
	 * Reads label index from a given path
	 * @param path path to label index
	 * @return label index map from label id to label name
	 */
    private Map<Integer, String> readLabelIndex(Path path) {
        return BayesUtils.readLabelIndex(conf, path);
    }

	/**
	 * Reads dictionary from a given path, which is a Hadoop sequence file
	 * @param path path to dictionary
	 * @return dictionary map from word to word id
	 */
    private Map<String, Integer> readDictionary(Path path) {
        Map<String, Integer> dict = new HashMap<>();
        String key;
        int value;
        for (Pair<Text, IntWritable> pair : new SequenceFileIterable<Text, IntWritable>(path, true, conf)) {
            key = pair.getFirst().toString();
            value = pair.getSecond().get();
            dict.put(key, value);
        }
        return dict;
    }

	/**
	 * Reads frequencies from a given path, which is a Hadoop sequence file
	 * @param path path to frequencies
	 * @return frequencies map from word id to frequency count
	 */
    private Map<Integer, Long> readFrequencies(Path path) {
        Map<Integer, Long> dict = new HashMap<>();
        int key;
        long value;
        for (Pair<IntWritable, LongWritable> pair : new SequenceFileIterable<IntWritable, LongWritable>(path, true, conf)) {
            key = pair.getFirst().get();
            value = pair.getSecond().get();
            dict.put(key, value);
        }
        return dict;
    }

	/**
	 * Returns analysis of instance
	 * @param instance text to analyze
	 * @return analysis of instance
	 * @throws IOException
	 */
	public Analysis analysis(String instance) throws IOException {
		throw new UnsupportedOperationException("should override");
	}

	/**
	 * Classifies instance
	 * @param instance text to analyze
	 * @return classification of instance to {@link classifier.spam.Label}
	 * @throws IOException
	 */
    public Label classify(String instance) throws IOException {
	    throw new UnsupportedOperationException("should override");
    }

}
