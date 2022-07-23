using Tensorflow;
using Tensorflow.Keras.Datasets;
using Tensorflow.Keras.Engine;
using Tensorflow.Keras.Losses;
using Tensorflow.Keras.Optimizers;
using Tensorflow.NumPy;

using static Tensorflow.Binding;
using static Tensorflow.KerasApi;

namespace _01_Tensorflow_ANNs
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading Mnist dataset...");
            Mnist? mnistDatasetLoader = new Mnist();
            DatasetPass? mnistDataset = mnistDatasetLoader.load_data();

            // Separate the dataset into features and labels.
            NDArray? trainDataX = mnistDataset.Train.Item1 / 255.0f;
            NDArray? trainDataY = mnistDataset.Train.Item2;

            // Separate the dataset into features and labels.
            NDArray? testDataX = mnistDataset.Test.Item1 / 255.0f;
            NDArray? testDataY = mnistDataset.Test.Item2;

            Console.WriteLine("Mnist dataset loaded!");

            Console.WriteLine("Creating the Model using the Functional API.");
            var inputLayer = keras.layers.Input(new Shape(28, 28), name: "Input Layer");
            var flattenLayer = keras.layers.Flatten().Apply(inputLayer);
            var denseLayer128 = keras.layers.Dense(128, activation: "relu").Apply(flattenLayer);
            var dropoutLayer = keras.layers.Dropout(0.2f).Apply(denseLayer128);
            var outputLayer = keras.layers.Dense(10, activation: "softmax").Apply(dropoutLayer);
            Functional model = keras.Model(inputLayer, outputLayer, name: "Simple Mnist dataset");
            Console.WriteLine("Model Created...");

            Console.WriteLine("Compiling model...");
            model.compile(new Adam(), new SparseCategoricalCrossentropy(), new string[] { "accuracy" });
            Console.WriteLine("Fitting model...");
            model.fit(trainDataX, trainDataY, batch_size: 32, epochs: 10, verbose: 1, use_multiprocessing: true);
            Console.WriteLine("Printing the model summary...");
            model.summary();

            Console.WriteLine("Evaluating model...");
            model.evaluate(testDataX, testDataY);
        }
    }
}