import React from "react";
import { Settings, HelpCircle } from "lucide-react";

export default function Header() {
  return (
    <header className="w-full bg-white border-b">
      <div className="max-w-7xl mx-auto px-1 sm:px-6 lg:px-8 py-1">
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-4">
            <img src="/seam-logo.png" alt="SEAM" className="h-8" />
            <img src="/new-logo.png" alt="SEAM" className="h-8" />
            <h1 className="text-xl font-bold text-gray-900">
              Face Authentication
            </h1>
          </div>

          <div className="flex items-start space-x-4">
            <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
              <Settings className="w-5 h-5" />
            </button>
            <button className="p-2 text-gray-500 hover:text-gray-700 rounded-full hover:bg-gray-100">
              <HelpCircle className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
