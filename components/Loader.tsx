
import React from 'react';

interface LoaderProps {
  message: string;
}

const Loader: React.FC<LoaderProps> = ({ message }) => {
  return (
    <div className="flex flex-col items-center justify-center h-full space-y-6 text-white">
      <div className="w-16 h-16 border-4 border-indigo-400 border-t-transparent border-solid rounded-full animate-spin"></div>
      <div className="text-center">
        <p className="text-2xl font-bold tracking-wider">Generating Proposal</p>
        <p className="text-lg text-gray-400 mt-2 animate-pulse">{message}</p>
      </div>
    </div>
  );
};

export default Loader;
