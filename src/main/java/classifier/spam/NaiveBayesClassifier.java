package classifier.spam;

import classifier.Classifier;
import classifier.analyses.Analysis;
import classifier.analyses.EnglishAnalysis;
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
import org.apache.mahout.math.Vector;
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
    private final StandardNaiveBayesClassifier classifier;
    private final Map<Integer, String> labels;
    private final Map<String, Integer> dictionary;
    private final Map<Integer, Long> frequencies;

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

    private Properties loadProperties(String name) throws IOException {
        Properties properties = new Properties();
        ClassLoader loader = Thread.currentThread().getContextClassLoader();
        try (InputStream stream = loader.getResourceAsStream(name)) {
            properties.load(stream);
	        LOGGER.info("loaded properties from " + name);
            return properties;
        }
    }

    private Map<Integer, String> readLabelIndex(Path path) {
        return BayesUtils.readLabelIndex(conf, path);
    }

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

    public Label classify(String instance) throws IOException {
	    Analysis analysis = new EnglishAnalysis(instance, dictionary, frequencies);
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
