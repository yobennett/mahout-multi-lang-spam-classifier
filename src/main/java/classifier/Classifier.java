package classifier;

import classifier.analyses.Analysis;

import java.io.IOException;

public interface Classifier<T, V> {

	T classify(V input) throws IOException;

	Analysis analysis(V input) throws IOException;

}
