import React, { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";

// 1. Define the Custom Normalization Layer with trainable weights
class NormalizationLayer extends tf.layers.Layer {
  mean: tf.Variable;
  std: tf.Variable;

  constructor(config: { mean?: number; std?: number } = {}) {
    super(config);
    // Initialize trainable variables instead of plain numbers
    this.mean = tf.variable(
      tf.scalar(config.mean ?? 0),
      true,
      "normalization/mean"
    );
    this.std = tf.variable(
      tf.scalar(config.std ?? 1),
      true,
      "normalization/std"
    );
  }

  build(inputShape: tf.Shape | tf.Shape[]) {
    // Optional: You can initialize weights here if needed
    super.build(inputShape);
  }

  call(inputs: tf.Tensor | tf.Tensor[], kwargs?: any) {
    return tf.tidy(() => {
      const inputTensor = Array.isArray(inputs) ? inputs[0] : inputs;
      return tf.div(tf.sub(inputTensor, this.mean), this.std);
    });
  }

  getConfig() {
    const baseConfig = super.getConfig();
    return {
      ...baseConfig,
      mean: this.mean.dataSync()[0],
      std: this.std.dataSync()[0],
    };
  }

  static get className() {
    return "Normalization";
  }
}

// Register the custom Normalization layer
tf.serialization.registerClass(NormalizationLayer);

// 2. Custom Hook to Load the Model with proper error handling
const useModel = (modelUrl: string) => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setLoading(true);
        console.log("Loading model from:", modelUrl);

        // Load the model with custom objects
        const loadedModel = await tf.loadLayersModel(modelUrl, {
          strict: false, // Allow missing weights
          customObjects: { Normalization: NormalizationLayer },
        });

        // Debug and initialize layers
        loadedModel.layers.forEach((layer) => {
          console.log(
            `Layer Name: ${layer.name}, Type: ${layer.getClassName()}`
          );
          if (layer instanceof NormalizationLayer) {
            console.log(`Normalization Layer initialized with default values`);
          }
        });

        await loadedModel.compile({
          optimizer: "adam",
          loss: "meanSquaredError",
          metrics: ["accuracy"],
        });

        setModel(loadedModel);
        console.log("Model loaded successfully.");
        loadedModel.summary();
      } catch (err: any) {
        console.error("Error loading model:", err);
        setError(`Model failed to load: ${err.message}`);
      } finally {
        setLoading(false);
      }
    };

    loadModel();

    // Cleanup function
    return () => {
      if (model) {
        model.dispose();
      }
    };
  }, [modelUrl]);

  return { model, loading, error };
};

// 3. Main Component with better error handling and memory management
const ModelLoader: React.FC = () => {
  const modelUrl = "/web_model/model.json";
  const { model, loading, error } = useModel(modelUrl);
  const [prediction, setPrediction] = useState<number[] | null>(null);

  const handlePrediction = async () => {
    if (!model) return;

    try {
      tf.tidy(() => {
        // Example: Perform a prediction with dummy input
        const dummyInput = tf.randomNormal([1, 55, 47, 3]);
        const predictionTensor = model.predict(dummyInput) as tf.Tensor;

        // Convert prediction to array and update state
        predictionTensor.array().then((predArray) => {
          setPrediction(Array.isArray(predArray) ? predArray[0] : []);
        });
      });
    } catch (err: any) {
      console.error("Error during prediction:", err);
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">TensorFlow.js Model Loader</h1>

      {loading && (
        <div className="bg-blue-100 p-4 rounded">
          <p>Loading model...</p>
        </div>
      )}

      {error && (
        <div className="bg-red-100 p-4 rounded">
          <p className="text-red-700">Error: {error}</p>
        </div>
      )}

      {!loading && !error && model && (
        <div className="space-y-4">
          <p className="text-green-600">Model loaded successfully!</p>
          <button
            onClick={handlePrediction}
            className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600"
          >
            Run Prediction
          </button>

          {prediction && (
            <div className="mt-4">
              <h2 className="text-xl font-semibold">Prediction Result:</h2>
              <pre className="bg-gray-100 p-4 rounded mt-2">
                {JSON.stringify(prediction, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ModelLoader;
