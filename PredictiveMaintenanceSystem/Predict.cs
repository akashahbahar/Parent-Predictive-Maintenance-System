using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace PredictiveMaintenanceSystem
{
    // === Input schema must match training schema (except label column) ===
    public class ModelInput
    {
        public float MachineID { get; set; }
        public string Timestamp { get; set; }
        public float Temperature_last { get; set; }
        public float Vibration_last { get; set; }
        public float Pressure_last { get; set; }
        public float RunTimeHours_last { get; set; }
        public float Temperature_mean { get; set; }
        public float Temperature_std { get; set; }
        public float Vibration_mean { get; set; }
        public float Vibration_std { get; set; }
        public float Vibration_rms { get; set; }
        public float Vibration_spec_energy { get; set; }
        public float Pressure_mean { get; set; }
        public float Pressure_std { get; set; }
    }

    // === Prediction schema ===
    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool WillFailWithin2h { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }

    class Inference
    {
        static void Main(string[] args)
        {
            string modelPath = Path.Combine(Environment.CurrentDirectory, "model.zip");

            MLContext mlContext = new MLContext();

            // Load trained model
            DataViewSchema modelSchema;
            ITransformer trainedModel = mlContext.Model.Load(modelPath, out modelSchema);

            var predEngine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(trainedModel);

            // === Simulated real-time sensor input ===
            var newData = new ModelInput
            {
                MachineID = 1,
                Timestamp = DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss"),
                Temperature_last = 82.4f,
                Vibration_last = 0.34f,
                Pressure_last = 31.2f,
                RunTimeHours_last = 245f,
                Temperature_mean = 80.0f,
                Temperature_std = 1.2f,
                Vibration_mean = 0.31f,
                Vibration_std = 0.05f,
                Vibration_rms = 0.33f,
                Vibration_spec_energy = 12.3f,
                Pressure_mean = 30.8f,
                Pressure_std = 0.6f
            };

            // Make prediction
            var prediction = predEngine.Predict(newData);

            Console.WriteLine("====== Prediction Result ======");
            Console.WriteLine($"MachineID: {newData.MachineID}");
            Console.WriteLine($"Timestamp: {newData.Timestamp}");
            Console.WriteLine($"Predicted Failure: {prediction.WillFailWithin2h}");
            Console.WriteLine($"Probability: {prediction.Probability:P2}");
            Console.WriteLine($"Raw Score: {prediction.Score}");
            Console.WriteLine("===============================");

            Console.WriteLine("Press any key to exit...");
            Console.ReadKey();
        }
    }
}
