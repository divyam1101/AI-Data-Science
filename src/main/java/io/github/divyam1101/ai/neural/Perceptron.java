package io.github.divyam1101.ai.neural;

/**
 * A single-layer perceptron binary classifier with a learnable bias, trained with the classic
 * perceptron learning rule.
 *
 * <p>Training stops once a full pass over the data makes no mistakes, which is guaranteed to
 * happen for linearly separable data (for example AND or OR, but not XOR).
 */
public final class Perceptron {

    private final double[] weights;
    private double bias;

    public Perceptron(int features) {
        this.weights = new double[features];
    }

    /**
     * @return the number of epochs needed to converge, or -1 if it did not converge within
     *         {@code maxEpochs}
     */
    public int train(double[][] inputs, int[] targets, double learningRate, int maxEpochs) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("inputs and targets differ in length");
        }
        for (int epoch = 1; epoch <= maxEpochs; epoch++) {
            int mistakes = 0;
            for (int n = 0; n < inputs.length; n++) {
                int error = targets[n] - predict(inputs[n]);
                if (error != 0) {
                    mistakes++;
                    for (int k = 0; k < weights.length; k++) {
                        weights[k] += learningRate * error * inputs[n][k];
                    }
                    bias += learningRate * error;
                }
            }
            if (mistakes == 0) {
                return epoch;
            }
        }
        return -1;
    }

    public int predict(double[] input) {
        double activation = bias;
        for (int k = 0; k < weights.length; k++) {
            activation += weights[k] * input[k];
        }
        return activation > 0 ? 1 : 0;
    }

    public double[] weights() {
        return weights.clone();
    }

    public double bias() {
        return bias;
    }
}
