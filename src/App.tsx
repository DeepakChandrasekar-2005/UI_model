import { CircuitBoard } from "lucide-react";
import FaceAuth from "./components/FaceAuth"; // Importing FaceAuth component

function App() {
  return (
    <div className="min-h-screen bg-white text-gray-900">
      <div className="max-w-4xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="text-start mb-6">
            {/* Logo Section */}
            <div className="flex items-center justify-between mb-1">
              {/* Left-aligned SEAM logo */}
              <img
                src="/SARZ LOGO.png"
                alt="SEAM"
                className="h-24 inline-block"
              />
              <img
                src="/seam-logo.jpeg"
                alt="SEAM"
                className="h-24 inline-block"
              />

              {/* Right-aligned new logo */}
              <img
                src="/new-logo.png"
                alt="New Logo"
                className="h-24 inline-block"
              />
            </div>

            {/* Title Section */}
            <div className="flex items-center justify-center gap-2 text-gray-600 mt-4">
              <CircuitBoard className="w-5 h-8" />
              <p className="text-lg">
                Secure Encryption and Authentication Model
              </p>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <FaceAuth />

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500">
          <p>
            Please ensure good lighting and position your face within the frame.
          </p>
          <p className="mt-2">
            © {new Date().getFullYear()} SEAM Authentication System
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
