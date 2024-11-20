import React from "react";

interface AuthStatusProps {
  status: "idle" | "processing" | "success" | "error";
  message: string;
}

export const AuthStatus: React.FC<AuthStatusProps> = ({ status, message }) => {
  const getStatusColor = () => {
    switch (status) {
      case "success":
        return "text-green-500";
      case "error":
        return "text-red-500";
      case "processing":
        return "text-blue-500";
      default:
        return "text-gray-500";
    }
  };

  return (
    <div className={`text-center ${getStatusColor()}`}>
      <p>{message}</p>
    </div>
  );
};
