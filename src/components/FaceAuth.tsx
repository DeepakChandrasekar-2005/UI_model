import React, { useState } from "react";
import useModel from "../hooks/useModel"; // Adjust the path to your `useModel` hook
import * as tf from "@tensorflow/tfjs";
import { CameraView } from "./CameraView"; // Ensure this path is correct
import { AuthStatus } from "./AuthStatus"; // Ensure this path is correct

const FaceAuth: React.FC = () => {
  const [authStatus, setAuthStatus] = useState<
    "idle" | "processing" | "success" | "error"
  >("idle");
  const [statusMessage, setStatusMessage] = useState(
    "Ready for authentication"
  );

  // Use the custom hook to load the model
  const { model, loading, error } = useModel("/web_model/model.json");

  const handleAuthenticate = async (imageSrc: string) => {
    if (loading) {
      setAuthStatus("error");
      setStatusMessage("Model is still loading. Please wait...");
      return;
    }

    if (error) {
      setAuthStatus("error");
      setStatusMessage("Model failed to load. Please try again.");
      return;
    }

    if (!model) {
      setAuthStatus("error");
      setStatusMessage("Model is not loaded. Please try again.");
      return;
    }

    setAuthStatus("processing");
    setStatusMessage("Processing authentication...");

    try {
      const image = new Image();
      image.src = imageSrc;
      await new Promise((resolve) => (image.onload = resolve));

      const tensor = tf.browser
        .fromPixels(image)
        .resizeBilinear([224, 224]) // Adjust size to match your model's input
        .expandDims(0)
        .div(255.0);

      const prediction = model.predict(tensor) as tf.Tensor;
      const result = await prediction.data();

      if (result[0] > 0.5) {
        setAuthStatus("success");
        setStatusMessage("Authentication successful!");
      } else {
        setAuthStatus("error");
        setStatusMessage("Authentication failed. Please try again.");
      }
    } catch (error) {
      console.error("Error during authentication:", error);
      setAuthStatus("error");
      setStatusMessage("An error occurred. Please try again.");
    }
  };

  return (
    <div className="bg-gray-100 rounded-xl shadow-lg p-8 mb-8 border border-gray-200">
      <div className="max-w-md mx-auto space-y-6">
        <CameraView onCapture={handleAuthenticate} />
        <AuthStatus status={authStatus} message={statusMessage} />
      </div>
    </div>
  );
};

export default FaceAuth;
