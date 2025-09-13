using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;

namespace PredictiveMaintenanceTrainer
{
    // === Data schema ===
    public class ModelInput
    {
        // Key identifiers
        [LoadColumn(0)] public float MachineID; // sometimes CSV will have MachineID as int; ML.NET expects floats for numeric
        [LoadColumn(1)] public string Timestamp; // keep as string if present

        // Raw last-sample columns (examples; adapt indexes to your CSV)
        // Update these LoadColumn indexes to match your CSV column order.
        [LoadColumn(2)] public float Temperature_last;
        [LoadColumn(3)] public float Vibration_last;
        [LoadColumn(4)] public float Pressure_last;
        [LoadColumn(5)] public float RunTimeHours_last;

        // Example engineered features — you might have many more; add as needed
        [LoadColumn(6)] public float Temperature_mean;
        [LoadColumn(7)] public float Temperature_std;
        [LoadColumn(8)] public float Vibration_mean;
        [LoadColumn(9)] public float Vibration_std;
        [LoadColumn(10)] public float Vibration_rms;
        [LoadColumn(11)] public float Vibration_spec_energy;
        [LoadColumn(12)] public float Pressure_mean;
        [LoadColumn(13)] public float Pressure_std;

        // Labels (one of these — depending on task)
        // Regression label (RUL in hours)
        // [LoadColumn(14), ColumnName("Label")] public float time_to_failure_hours;

        // If using classification instead, ensure the Label column is 0/1 for the chosen binary label.
        // e.g. if your CSV has will_fail_within_2h column, change LoadColumn index and uncomment next line:
        [LoadColumn(15), ColumnName("Label")] public bool will_fail_within_2h;
    }

    // For regression prediction result
    public class RegressionPrediction
    {
        [ColumnName("Score")]
        public float PredictedRUL;
    }

    // For binary classification result
    public class BinaryPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabel;
        public float Score;
        public float Probability;
    }

    class Program
    {
        // Adjust this to your CSV path
        private static string DataPath = Path.Combine(Environment.CurrentDirectory, "features_engineered.csv");
        private static string ModelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");

        static void Main(string[] args)
        {
            // Choose mode: "regression" or "classification"
            // By default we train regression on time_to_failure_hours. Change to "classification" if you prefer.
            string mode = args.Length > 0 ? args[0].ToLower() : "classification";
            Console.WriteLine($"Mode: {mode}");
            var mlContext = new MLContext(seed: 0);

            if (!File.Exists(DataPath))
            {
                Console.WriteLine($"ERROR: Data file not found at {DataPath}");
                return;
            }

            // Load data
            var textLoaderOptions = new TextLoader.Options
            {
                Separators = new[] { ',' },
                HasHeader = true,
                AllowQuoting = true,
                Columns = new[]
                {
                    // Define columns by name & index mapping here for robust loading.
                    // If your CSV has headers, you can use LoadFromTextFile<ModelInput>(path, hasHeader:true) instead.
                    new TextLoader.Column("MachineID", DataKind.Single, 35),
                    new TextLoader.Column("Timestamp", DataKind.String, 34),
                    new TextLoader.Column("Temperature_last", DataKind.Single, 36),
                    new TextLoader.Column("Vibration_last", DataKind.Single, 37),
                    new TextLoader.Column("Pressure_last", DataKind.Single, 38),
                    new TextLoader.Column("RunTimeHours_last", DataKind.Single, 39),
                    new TextLoader.Column("Temperature_mean", DataKind.Single, 0),
                    new TextLoader.Column("Temperature_std", DataKind.Single, 1),
                    new TextLoader.Column("Vibration_mean", DataKind.Single, 8),
                    new TextLoader.Column("Vibration_std", DataKind.Single, 9),
                    new TextLoader.Column("Vibration_rms", DataKind.Single, 32),
                    new TextLoader.Column("Vibration_spec_energy", DataKind.Single, 33),
                    new TextLoader.Column("Pressure_mean", DataKind.Single, 16),
                    new TextLoader.Column("Pressure_std", DataKind.Single, 17),
                    // new TextLoader.Column("time_to_failure_hours", DataKind.Single, 41),
                    // if you want classification label instead, add another column mapping
                    new TextLoader.Column("will_fail_within_2h", DataKind.Boolean, 42),
                }
            };
            var loader = mlContext.Data.CreateTextLoader(textLoaderOptions);
            var data = loader.Load(DataPath);

            // OPTIONAL: Inspect column schema
            Console.WriteLine("Schema:");
            foreach (var col in data.Schema)
                Console.WriteLine($"  {col.Name} - {col.Type}");

            // Data split: important — avoid leakage.
            // Simple approach: time-based split per machine is ideal, but here we'll do TrainTestSplit and warn the user.
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 0);
            var trainData = split.TrainSet;
            var testData = split.TestSet;

            trainData = mlContext.Transforms.CopyColumns("Label", "will_fail_within_2h").Fit(trainData).Transform(trainData);
            testData = mlContext.Transforms.CopyColumns("Label", "will_fail_within_2h").Fit(testData).Transform(testData);

            if (mode == "regression")
            {
                TrainRegression(mlContext, trainData, testData);
            }
            else if (mode == "classification")
            {
                TrainBinaryClassification(mlContext, trainData, testData);
            }
            else
            {
                Console.WriteLine("Unknown mode. Use 'regression' or 'classification'");
            }
        }

        static void TrainRegression(MLContext mlContext, IDataView trainData, IDataView testData)
        {
            Console.WriteLine("Building regression pipeline (RUL prediction)...");

            // Define features (exclude Label and non-numeric columns like Timestamp)
            var featureColumns = new[]
            {
                "MachineID",
                "Temperature_last","Vibration_last","Pressure_last","RunTimeHours_last",
                "Temperature_mean","Temperature_std",
                "Vibration_mean","Vibration_std","Vibration_rms","Vibration_spec_energy",
                "Pressure_mean","Pressure_std"
            };

            var pipeline = mlContext.Transforms.ReplaceMissingValues(featureColumns.Select(col => new InputOutputColumnPair(col, col)).ToArray(), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.Concatenate("Features", featureColumns))
                .Append(mlContext.Transforms.NormalizeMeanVariance("Features"))
                // You can choose different trainers:
                .Append(mlContext.Regression.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                ;

            Console.WriteLine("Training regression model...");
            var model = pipeline.Fit(trainData);

            Console.WriteLine("Evaluating on test set...");
            var preds = model.Transform(testData);
            var metrics = mlContext.Regression.Evaluate(preds, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"  RMSE: {metrics.RootMeanSquaredError:F4}");
            Console.WriteLine($"  MAE : {metrics.MeanAbsoluteError:F4}");
            Console.WriteLine($"  R^2 : {metrics.RSquared:F4}");

            Console.WriteLine($"Saving model to {ModelPath} ...");
            mlContext.Model.Save(model, trainData.Schema, ModelPath);
            Console.WriteLine("Model saved.");

            // Example prediction
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, RegressionPrediction>(model);
            var sample = new ModelInput
            {
                MachineID = 1,
                Temperature_last = 78,
                Vibration_last = 0.55f,
                Pressure_last = 5.1f,
                RunTimeHours_last = 1200,
                Temperature_mean = 76,
                Temperature_std = 2.5f,
                Vibration_mean = 0.5f,
                Vibration_std = 0.1f,
                Vibration_rms = 0.51f,
                Vibration_spec_energy = 0.002f,
                Pressure_mean = 5.0f,
                Pressure_std = 0.3f
            };
            var r = predEngine.Predict(sample);
            Console.WriteLine($"Sample predicted RUL (hours): {r.PredictedRUL:F4}");
        }

        static void TrainBinaryClassification(MLContext mlContext, IDataView trainData, IDataView testData)
        {
            Console.WriteLine("Building binary classification pipeline (will_fail_within_Nh)...");

            // If your CSV's Label column is boolean (true/false), ensure it's named "Label".
            var featureColumns = new[]
            {
                "MachineID",
                "Temperature_last","Vibration_last","Pressure_last","RunTimeHours_last",
                "Temperature_mean","Temperature_std",
                "Vibration_mean","Vibration_std","Vibration_rms","Vibration_spec_energy",
                "Pressure_mean","Pressure_std"
            };

            var pipeline = mlContext.Transforms.ReplaceMissingValues(featureColumns.Select(col=> new InputOutputColumnPair(col, col)).ToArray(), replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean)
                .Append(mlContext.Transforms.Concatenate("Features", featureColumns))
                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                // Example: LightGbm binary classifier (tune parameters in practice)
                .Append(mlContext.BinaryClassification.Trainers.LightGbm(labelColumnName: "Label", featureColumnName: "Features"))
                ;

            Console.WriteLine("Training classification model...");
            var model = pipeline.Fit(trainData);

            Console.WriteLine("Evaluating...");
            var preds = model.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(preds, labelColumnName: "Label", scoreColumnName: "Score");
            Console.WriteLine($"  Accuracy: {metrics.Accuracy:P2}");
            Console.WriteLine($"  AUC     : {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  F1Score : {metrics.F1Score:P2}");

            Console.WriteLine($"Saving model to {ModelPath} ...");
            mlContext.Model.Save(model, trainData.Schema, ModelPath);
            Console.WriteLine("Model saved.");

            // Example prediction
            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, BinaryPrediction>(model);
            var sample = new ModelInput
            {
                MachineID = 2,
                Temperature_last = 92,
                Vibration_last = 0.72f,
                Pressure_last = 6.5f,
                RunTimeHours_last = 2500,
                Temperature_mean = 88,
                Temperature_std = 1.8f,
                Vibration_mean = 0.7f,
                Vibration_std = 0.09f,
                Vibration_rms = 0.71f,
                Vibration_spec_energy = 0.010f,
                Pressure_mean = 6.3f,
                Pressure_std = 0.4f
            };
            var r = predEngine.Predict(sample);
            Console.WriteLine($"Sample predicted will-fail: {r.PredictedLabel} (prob={r.Probability:P2})");
        }
    }
}
