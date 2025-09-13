using Microsoft.ML;
using System;
using System.Threading;

class InferenceApp
{
    static void Main(string[] args)
    {
        MLContext mlContext = new MLContext();

        // Load model
        DataViewSchema modelSchema;
        ITransformer trainedModel = mlContext.Model.Load("model.zip", out modelSchema);

        // Create prediction engine
        var predEngine = mlContext.Model.CreatePredictionEngine<SensorData, FailurePrediction>(trainedModel);

        // Simulate real-time predictions
        Random rnd = new Random();
        while (true)
        {
            var newData = new SensorData
            {
                MachineID = 1,
                Temperature_last = (float)(60 + rnd.NextDouble() * 50),
                Vibration_last = (float)(0.1 + rnd.NextDouble()),
                Pressure_last = (float)(100 + rnd.NextDouble() * 20),
                RunTimeHours_last = (float)(rnd.Next(1, 1000)),

                Temperature_mean = (float)(70 + rnd.NextDouble() * 10),
                Temperature_std = (float)(rnd.NextDouble() * 5),
                Vibration_mean = (float)(0.3 + rnd.NextDouble() * 0.2),
                Vibration_std = (float)(rnd.NextDouble() * 0.1),
                Vibration_rms = (float)(rnd.NextDouble()),
                Vibration_spec_energy = (float)(rnd.NextDouble()),
                Pressure_mean = (float)(110 + rnd.NextDouble() * 5),
                Pressure_std = (float)(rnd.NextDouble() * 2)
            };

            // Predict
            var prediction = predEngine.Predict(newData);

            Console.WriteLine(
                $"Machine {newData.MachineID} | Temp: {newData.Temperature_last:F1} | Vib: {newData.Vibration_last:F2} | " +
                $"Failure: {prediction.WillFailWithin2h} (Prob={prediction.Probability:P2})");

            Thread.Sleep(2000); // wait 2 seconds before next reading
        }
    }
}

// Input schema (must match training schema)
public class SensorData
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

// Output schema
public class FailurePrediction
{
    public bool WillFailWithin2h { get; set; }
    public float Probability { get; set; }
}
