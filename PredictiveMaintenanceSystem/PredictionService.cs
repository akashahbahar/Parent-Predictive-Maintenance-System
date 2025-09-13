using Microsoft.ML;
using System;

namespace PredictiveMaintenanceSystem
{
    public class PredictionService
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;
        private readonly PredictionEngine<ModelInput, ModelOutput> _engine;

        public PredictionService()
        {
            _mlContext = new MLContext();
            _model = _mlContext.Model.Load("model.zip", out _);
            _engine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
        }

        public ModelOutput Predict(ModelInput input)
        {
            return _engine.Predict(input);
        }
    }
}