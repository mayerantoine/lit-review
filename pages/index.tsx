"use client"

import { useState, FormEvent, useRef, DragEvent, ChangeEvent } from 'react';
import { useRouter } from 'next/router';

interface IndexStats {
  total_abstracts: number;
  chunks_created: number;
  total_indexed: number;
}

export default function Home() {
  const [researchIdea, setResearchIdea] = useState('');
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [fileError, setFileError] = useState<string>('');
  const [isIndexing, setIsIndexing] = useState(false);
  const [indexStats, setIndexStats] = useState<IndexStats | null>(null);
  const [indexError, setIndexError] = useState<string>('');
  const fileInputRef = useRef<HTMLInputElement>(null);
  const router = useRouter();

  //const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  const validateFile = (file: File): boolean => {
    // Reset error
    setFileError('');

    // Check if it's a CSV file
    if (!file.name.toLowerCase().endsWith('.csv')) {
      setFileError('Please upload a CSV file');
      return false;
    }

    // Check file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setFileError('File size must be less than 10MB');
      return false;
    }

    return true;
  };

  const handleFile = (file: File) => {
    if (validateFile(file)) {
      setUploadedFile(file);
      setFileError('');
    }
  };

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      handleFile(files[0]);
    }
  };

  const handleBrowseClick = () => {
    fileInputRef.current?.click();
  };

  const handleRemoveFile = () => {
    setUploadedFile(null);
    setFileError('');
    setIndexStats(null);
    setIndexError('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleUploadAndIndex = async () => {
    if (!uploadedFile) return;

    setIsIndexing(true);
    setIndexError('');

    const formData = new FormData();
    formData.append('file', uploadedFile);

    try {
      const response = await fetch(`/api/upload-and-index`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Failed to index file');
      }

      const result = await response.json();
      setIndexStats({
        total_abstracts: result.total_abstracts,
        chunks_created: result.chunks_created,
        total_indexed: result.total_indexed,
      });
    } catch (error) {
      setIndexError(error instanceof Error ? error.message : 'Failed to upload and index file');
    } finally {
      setIsIndexing(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();

    // Store research idea in sessionStorage
    sessionStorage.setItem('researchIdea', researchIdea);

    // Store file data if uploaded
    if (uploadedFile) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const csvContent = event.target?.result as string;
        sessionStorage.setItem('csvFile', csvContent);
        sessionStorage.setItem('csvFileName', uploadedFile.name);
        // Navigate to generate page
        router.push('/generate');
      };
      reader.readAsText(uploadedFile);
    } else {
      // Navigate without file
      router.push('/generate');
    }
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="flex items-center px-6 py-4 border-b border-black/10 dark:border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 text-primary">
            <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
              <path d="M42.4379 44C42.4379 44 36.0744 33.9038 41.1692 24C46.8624 12.9336 42.2078 4 42.2078 4L7.01134 4C7.01134 4 11.6577 12.932 5.96912 23.9969C0.876273 33.9029 7.27094 44 7.27094 44L42.4379 44Z" fill="currentColor"></path>
            </svg>
          </div>
          <h1 className="text-lg font-bold text-black dark:text-white">
            ResearchAI
          </h1>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-grow flex items-center justify-center p-6">
        <div className="w-full max-w-2xl mx-auto space-y-8">
          {/* Hero Section */}
          <div className="text-center">
            <h2 className="text-3xl font-bold text-black dark:text-white">
              Generate Related Work
            </h2>
            <p className="mt-2 text-black/60 dark:text-white/60">
              Start by providing your research idea and a list of references.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-6">
            {/* Research Idea Input */}
            <div>
              <label
                className="block text-sm font-medium text-black dark:text-white mb-2"
                htmlFor="research-idea"
              >
                Your Research Idea
              </label>
              <textarea
                className="w-full h-36 p-4 rounded-lg bg-white dark:bg-black/20 border border-black/10 dark:border-white/10 focus:ring-2 focus:ring-primary focus:border-primary transition duration-200 resize-none text-black dark:text-white placeholder:text-black/40 dark:placeholder:text-white/40"
                id="research-idea"
                placeholder="e.g., 'Using large language models to summarize legal documents'"
                value={researchIdea}
                onChange={(e) => setResearchIdea(e.target.value)}
                required
              ></textarea>
            </div>

            {/* File Upload Area */}
            <div>
              <label className="block text-sm font-medium text-black dark:text-white mb-2">
                References (CSV File)
              </label>
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                className={`relative flex flex-col items-center justify-center p-8 border-2 border-dashed rounded-xl text-center transition-all ${
                  isDragging
                    ? 'border-primary bg-primary/5 dark:bg-primary/10'
                    : uploadedFile
                    ? 'border-primary/50 bg-primary/5 dark:bg-primary/10'
                    : 'border-black/20 dark:border-white/20 hover:border-primary/50'
                }`}
              >
                {!uploadedFile ? (
                  <>
                    <span className="material-symbols-outlined text-4xl text-black/40 dark:text-white/40 mb-4">
                      upload_file
                    </span>
                    <h3 className="text-lg font-semibold text-black dark:text-white">
                      {isDragging ? 'Drop your CSV file here' : 'Drag and drop a CSV file'}
                    </h3>
                    <p className="mt-1 text-sm text-black/60 dark:text-white/60">
                      The file should contain columns: id, title, abstract
                    </p>
                    <p className="mt-4 text-sm text-black/60 dark:text-white/60">or</p>
                    <button
                      type="button"
                      onClick={handleBrowseClick}
                      className="mt-4 px-6 py-2.5 text-sm font-semibold text-white rounded-lg transition-colors shadow-sm"
                      style={{ backgroundColor: '#1173d4' }}
                      onMouseEnter={(e) => (e.currentTarget.style.backgroundColor = '#0d5aa8')}
                      onMouseLeave={(e) => (e.currentTarget.style.backgroundColor = '#1173d4')}
                    >
                      Browse Files
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv"
                      onChange={handleFileInputChange}
                      className="hidden"
                    />
                  </>
                ) : (
                  <div className="w-full space-y-4">
                    <div className="flex items-center justify-between bg-white dark:bg-black/20 rounded-lg p-4 border border-primary/30">
                      <div className="flex items-center gap-3 flex-1 min-w-0">
                        <span className="material-symbols-outlined text-2xl text-primary flex-shrink-0">
                          description
                        </span>
                        <div className="flex-1 min-w-0 text-left">
                          <p className="text-sm font-medium text-black dark:text-white truncate">
                            {uploadedFile.name}
                          </p>
                          <p className="text-xs text-black/60 dark:text-white/60">
                            {formatFileSize(uploadedFile.size)}
                          </p>
                        </div>
                      </div>
                      <button
                        type="button"
                        onClick={handleRemoveFile}
                        disabled={isIndexing}
                        className="ml-3 p-2 text-black/60 dark:text-white/60 hover:text-red-500 dark:hover:text-red-400 transition-colors flex-shrink-0 disabled:opacity-50"
                        title="Remove file"
                      >
                        <span className="material-symbols-outlined text-xl">close</span>
                      </button>
                    </div>

                    {!indexStats && !isIndexing && (
                      <button
                        type="button"
                        onClick={handleUploadAndIndex}
                        disabled={isIndexing}
                        className="w-full px-6 py-3 text-sm font-semibold text-white rounded-lg transition-colors shadow-sm flex items-center justify-center gap-2"
                        style={{ backgroundColor: '#1173d4' }}
                        onMouseEnter={(e) => !isIndexing && (e.currentTarget.style.backgroundColor = '#0d5aa8')}
                        onMouseLeave={(e) => !isIndexing && (e.currentTarget.style.backgroundColor = '#1173d4')}
                      >
                        <span className="material-symbols-outlined">cloud_upload</span>
                        Upload & Index File
                      </button>
                    )}

                    {isIndexing && (
                      <div className="flex items-center justify-center gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                        <span className="text-sm font-medium text-primary">Indexing file...</span>
                      </div>
                    )}

                    {indexStats && (
                      <div className="p-4 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
                        <div className="flex items-start gap-2">
                          <span className="material-symbols-outlined text-green-600 dark:text-green-400 flex-shrink-0">check_circle</span>
                          <div className="flex-1">
                            <p className="text-sm font-semibold text-green-800 dark:text-green-200 mb-2">
                              Index created successfully!
                            </p>
                            <div className="space-y-1 text-xs text-green-700 dark:text-green-300">
                              <p>Papers indexed: {indexStats.total_abstracts}</p>
                              <p>Chunks created: {indexStats.chunks_created}</p>
                              <p>Total documents: {indexStats.total_indexed}</p>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={handleBrowseClick}
                      disabled={isIndexing}
                      className="text-sm text-primary hover:text-primary/80 font-medium disabled:opacity-50"
                    >
                      Choose a different file
                    </button>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv"
                      onChange={handleFileInputChange}
                      className="hidden"
                    />
                  </div>
                )}
              </div>
              {fileError && (
                <p className="mt-2 text-sm text-red-500 dark:text-red-400 flex items-center gap-1">
                  <span className="material-symbols-outlined text-base">error</span>
                  {fileError}
                </p>
              )}
              {indexError && (
                <p className="mt-2 text-sm text-red-500 dark:text-red-400 flex items-center gap-1">
                  <span className="material-symbols-outlined text-base">error</span>
                  {indexError}
                </p>
              )}
            </div>

            {/* Generate Button */}
            <div className="flex justify-end">
              <button
                type="submit"
                className="w-full sm:w-auto px-8 py-3 text-base font-bold text-white rounded-lg transition-colors flex items-center justify-center gap-2"
                style={{ backgroundColor: '#1173d4' }}
                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#0d5aa8'}
                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#1173d4'}
              >
                <span className="material-symbols-outlined">auto_awesome</span>
                Generate Related Work
              </button>
            </div>
          </form>
        </div>
      </main>
    </div>
  );
}
