using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Keras.PreProcessing.Image;

using Plotly.NET.CSharp;

using static Plotly.NET.GenericChart;

namespace ConvolutionalNeuralNetwork
{
    internal class Program
    {
        static void Main(string[] args)
        {
            #region Paths for files and folders(where the data is stored in the computer).
            string trainFolder = @".\Resources\Train";
            string testFolder = @".\Resources\Test";

            string trainPizzaFolder = @".\Resources\Train\Pizza";
            string trainSteakFolder = @".\Resources\Train\Steak";
            string testPizzaFolder = @".\Resources\Test\Pizza";
            string testSteakFolder = @".\Resources\Test\Steak";
            #endregion
            // Calculate how many files are found in each folder(that inclues Train and Test data).
            List<string> trainPizzaImages = PrintNumberOfFilesFound(trainPizzaFolder, true);
            List<string> trainSteakImages = PrintNumberOfFilesFound(trainSteakFolder, true);
            List<string> testPizzaImages = PrintNumberOfFilesFound(testPizzaFolder, false);
            List<string> testSteakImages = PrintNumberOfFilesFound(testSteakFolder, false);

            Console.WriteLine("Processing image Data into Tensors.");

            // Creating the Image Data Generator, which is used to preprocess the image data and turn it into Tensors.
            // Augument the Image Data to make the Neural Network better at generalization, reducing overfitting.
            // This augmentation can be in the form of multiple kinds of image transformation like rescale, random rotation
            // zoom, flip and others.
            ImageDataGenerator trainDataGenerator = new ImageDataGenerator(rescale: 1.0f / 255, rotation_range: 25,
                shear_range: 0.2f, zoom_range: 0.2f, width_shift_range: 0.2f, height_shift_range: 0.2f, horizontal_flip: true);
            ImageDataGenerator testDataGenerator = new ImageDataGenerator(rescale: 1.0f / 255);

            // Loading the Data into the Generator so we can get the data already transformed into Tensors.
            // Also separates data into Batches.
            KerasIterator trainData = trainDataGenerator.FlowFromDirectory(trainFolder, target_size: new Tuple<int, int>(224, 224),
                class_mode: "binary", batch_size: 32, seed: 42);
            KerasIterator testData = testDataGenerator.FlowFromDirectory(testFolder, target_size: new Tuple<int, int>(224, 224),
                class_mode: "binary", batch_size: 32, seed: 42);

            Console.WriteLine("Done preprocessing Image Data.");

            Console.WriteLine("Creating CNN Model...");

            // Values used to setup the Neural Network.
            int filters = 12;
            Tuple<int, int> kernelSize = new Tuple<int, int>(3, 3);
            Tuple<int, int> poolSize = new Tuple<int, int>(2, 2);

            // Creating the Neural Network Model and adding Hidden Layers.
            Sequential model = new Sequential();
            model.Add(new Conv2D(filters, kernelSize, activation: "relu", input_shape: (224, 224, 3)));
            model.Add(new Conv2D(filters, kernelSize, activation: "relu"));
            model.Add(new MaxPooling2D(poolSize, padding: "valid"));
            model.Add(new Conv2D(filters, kernelSize, activation: "relu"));
            model.Add(new Conv2D(filters, kernelSize, activation: "relu"));
            model.Add(new MaxPooling2D(poolSize));
            model.Add(new Flatten());
            model.Add(new Dense(1, activation: "sigmoid"));

            // Compile the model.
            model.Compile(new Adam(), loss: "binary_crossentropy", metrics: new string[] { "accuracy" });

            Console.WriteLine("Model created!");
            Console.WriteLine("Training model...");

            // Calculate the steps between batches based on amount of data.
            int trainSteps = (int)Math.Ceiling((trainPizzaImages.Count + trainSteakImages.Count) / 32.0d);
            int testSteps = (int)Math.Ceiling((testPizzaImages.Count + testSteakImages.Count) / 32.0d);

            Console.WriteLine($"Train Steps: {trainSteps}");
            Console.WriteLine($"Test Steps: {testSteps}");

            // Train(fit) model and put it into a history so we can Plot the loss and accuracy of Train and Test data.
            Keras.Callbacks.History? history1 = model.FitGenerator(trainData, epochs: 50, steps_per_epoch: trainSteps,
                validation_data: testData, validation_steps: testSteps, verbose: 1);
            #region Plotting the Loss curves of the Model.
            // Creating the chart to plot the Loss line of the Train and Test data.
            GenericChart lossLine = Chart.Line<int, double, string>(history1.Epoch, history1.HistoryLogs["loss"], Name: "Loss Chart", ShowLegend: true);
            GenericChart validationLossLine = Chart.Line<int, double, string>(history1.Epoch, history1.HistoryLogs["val_loss"], Name: "Validation Loss Chart", ShowLegend: true);

            // Creating a multi chart plot that combines the Loss lines.
            List<GenericChart> lossLines = new List<GenericChart>() { lossLine, validationLossLine };
            GenericChart combinedLossChart = Chart.Combine(lossLines);
            combinedLossChart.WithTraceInfo();
            combinedLossChart.WithXAxisStyle<int, int, string>(TitleText: "Epochs");
            combinedLossChart.WithYAxisStyle<double, double, string>(TitleText: "Loss");
            #endregion

            #region Plotting the Accuracy curves of the Model.
            // Creating the chart to plot the Loss line of the Train and Test data.
            GenericChart accuracyLine = Chart.Line<int, double, string>(history1.Epoch, history1.HistoryLogs["accuracy"], Name: "Accuracy Chart", ShowLegend: true);
            GenericChart validationaccuracyLine = Chart.Line<int, double, string>(history1.Epoch, history1.HistoryLogs["val_accuracy"], Name: "Validation Accuracy Chart", ShowLegend: true);

            // Creating a multi chart plot that combines the Loss lines.
            List<GenericChart> accuracyLines = new List<GenericChart>() { accuracyLine, validationaccuracyLine };
            GenericChart combinedAccuracyChart = Chart.Combine(accuracyLines);
            combinedLossChart.WithTraceInfo();
            combinedAccuracyChart.WithXAxisStyle<int, int, string>(TitleText: "Epochs");
            combinedAccuracyChart.WithYAxisStyle<double, double, string>(TitleText: "Accuracy");
            #endregion

            #region Combining both Loss and Accuracy combined charts into a single StackChart.
            GenericChart stackChart = Chart.Grid(new List<GenericChart>() { combinedLossChart, combinedAccuracyChart }, 2, 1);
            stackChart.Show();
            #endregion

            List<string> PrintNumberOfFilesFound(string folderPath, bool trainingSamples)
            {
                var parentDirectory = Directory.EnumerateFiles(folderPath);
                Console.WriteLine($"Found {parentDirectory.Count()} {folderPath.Split('\\').Last()} images for {(trainingSamples ? "training" : "test")}.");

                return parentDirectory.ToList();
            }
        }
    }
}