using Keras;
using Keras.Datasets;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;

using Numpy;

namespace _02_Tensorflow_CNNs
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("Loading Cifar10 Dataset...");
            ((NDarray, NDarray), (NDarray, NDarray)) cifarDataset = Cifar10.LoadData();

            NDarray? trainX = cifarDataset.Item1.Item1 / 255.0f;
            NDarray? trainY = cifarDataset.Item1.Item2;

            NDarray? testX = cifarDataset.Item2.Item1 / 255.0f;
            NDarray? testY = cifarDataset.Item2.Item2;

            Console.WriteLine("Building Model using Functional API...");
            Tuple<int, int> kernelSize = new Tuple<int, int>(3, 3);
            Tuple<int, int> poolSize = new Tuple<int, int>(2, 2);

            var inputs = new Input(new Shape(trainX[0].shape.Dimensions));
            var x = new Conv2D(32, kernel_size: kernelSize, activation: "relu", padding: "same").Set(inputs);
            x = new BatchNormalization().Set(x);
            x = new Conv2D(32, kernel_size: kernelSize, activation: "relu", padding: "same").Set(x);
            //x = new BatchNormalization().Set(x);
            x = new MaxPooling2D(poolSize).Set(x);
            x = new Conv2D(64, kernel_size: kernelSize, activation: "relu", padding: "same").Set(x);
            //x = new BatchNormalization().Set(x);
            x = new Conv2D(64, kernel_size: kernelSize, activation: "relu", padding: "same").Set(x);
            //x = new BatchNormalization().Set(x);
            x = new MaxPooling2D(poolSize).Set(x);
            x = new Conv2D(128, kernel_size: kernelSize, activation: "relu", padding: "same").Set(x);
            //x = new BatchNormalization().Set(x);
            x = new Conv2D(128, kernel_size: kernelSize, activation: "relu", padding: "same").Set(x);
            //x = new BatchNormalization().Set(x);
            x = new MaxPooling2D(poolSize).Set(x);
            x = new Flatten().Set(x);
            x = new Dropout(0.20f).Set(x);
            x = new Dense(256).Set(x);
            var output = new Dense(10, activation: "softmax").Set(x);

            Model model = new Model(new BaseLayer[] { inputs }, new BaseLayer[] { output });

            Console.WriteLine("Compiling model...");
            model.Compile(new Adam(), "sparse_categorical_crossentropy", new string[] { "accuracy" });

            Console.WriteLine("Training model...");
            model.Fit(trainX, trainY, batch_size: 16, epochs: 10, verbose: 2);

            Console.WriteLine("Evaluating model accuracy...");
            model.Evaluate(testX, testY);
        }
    }
}