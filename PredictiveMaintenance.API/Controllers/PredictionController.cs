using Microsoft.AspNetCore.Mvc;
using PredictiveMaintenanceAPI.Services;
using PredictiveMaintenanceSystem;
using System.Text.Json;

namespace PredictiveMaintenanceAPI.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class InferenceController : ControllerBase
    {
        private readonly PredictionService _predictionService;
        private readonly ILogger<InferenceController> _logger;
        private readonly NotificationService _notificationService;
        private readonly IConfiguration _config;

        public InferenceController(PredictionService predictionService, ILogger<InferenceController> logger, NotificationService notificationService, IConfiguration config)
        {
            _predictionService = predictionService;
            _logger = logger;
            _notificationService = notificationService;
            _config = config;
        }

        [HttpPost]
        public ActionResult<ModelOutput> Post([FromBody] ModelInput input)
        {
            var inputJson = JsonSerializer.Serialize(input);
            _logger.LogInformation("Received Inference Request: {@Input}", inputJson);
            
            var prediction = _predictionService.Predict(input);

            double threshold = _config.GetValue<double>("AlertSettings:FailureThreshold");
            if (prediction.Probability > threshold)
            {
                string alertMessage = $@"
                🚨 Alert: Machine failure probability is {prediction.Probability:P2}!

                Machine ID: {input.MachineID}
                Timestamp: {input.Timestamp}
                Temperature (last): {input.Temperature_last}
                Vibration (last): {input.Vibration_last}
                Pressure (last): {input.Pressure_last}
                Run Time (hours, last): {input.RunTimeHours_last}

                Temperature (mean): {input.Temperature_mean}
                Temperature (std): {input.Temperature_std}
                Vibration (mean): {input.Vibration_mean}
                Vibration (std): {input.Vibration_std}
                Vibration (rms): {input.Vibration_rms}
                Vibration (spec energy): {input.Vibration_spec_energy}
                Pressure (mean): {input.Pressure_mean}
                Pressure (std): {input.Pressure_std}

                Prediction: {(prediction.WillFailWithin2h ? "Likely to fail within 2 hours" : "Unlikely to fail within 2 hours")}
                Raw Score: {prediction.Score:F4}
                ";

                // Send Email
                _notificationService.SendEmailAlert("Critical Machine Alert", alertMessage);

            }

            return Ok(prediction);
        }
    }
}
