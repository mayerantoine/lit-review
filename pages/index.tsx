import { useState, FormEvent } from 'react';
import { useRouter } from 'next/router';

export default function Home() {
  const [researchIdea, setResearchIdea] = useState('');
  const router = useRouter();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    // Store research idea in sessionStorage
    sessionStorage.setItem('researchIdea', researchIdea);
    // Navigate to generate page
    router.push('/generate');
  };

  return (
    <div className="flex flex-col min-h-screen">
      {/* Header */}
      <header className="flex items-center justify-between px-6 py-4 border-b border-black/10 dark:border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 flex items-center justify-center text-white bg-primary rounded-lg">
            <span className="material-symbols-outlined">science</span>
          </div>
          <h1 className="text-lg font-bold text-black dark:text-white">
            Research Assistant
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <nav className="hidden md:flex items-center gap-6 text-sm font-medium text-black/60 dark:text-white/60">
            <a className="hover:text-primary" href="#">
              Home
            </a>
            <a className="hover:text-primary" href="#">
              My Library
            </a>
            <a className="hover:text-primary" href="#">
              Explore
            </a>
            <a className="hover:text-primary" href="#">
              Feed
            </a>
          </nav>
          <div className="flex items-center gap-4">
            <button className="relative text-black/60 dark:text-white/60 hover:text-primary">
              <span className="material-symbols-outlined">notifications</span>
              <div className="absolute -top-1 -right-1 w-2 h-2 bg-primary rounded-full"></div>
            </button>
            <div
              className="w-10 h-10 rounded-full bg-cover bg-center"
              style={{
                backgroundImage:
                  'url("https://lh3.googleusercontent.com/aida-public/AB6AXuAyQcDmm4stFW2M0PVcHFVpQyyHPU0llSFKzrL4A44_-zEiYxX5d526Mjru5YuB8DAulqRDkA10UQSXAWfUcKF-ZLRCV-LlGAMmWjR2GRQ4yv25I544TZ1MhkvezG6IxKYZbNHbT8lS6ssrKDqq-YKnU6CCyfooQ7F1ReRogGx04wXCCt6rTzWzyWoipwrt_S_vYR-GmncOqQUspFTj5Ms1yWhbrD1fpTVz5Lf2PUDlD7mMA5qaHcFO6GbGhCvGgc9RkkjdZfglAVM")',
              }}
            ></div>
          </div>
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
            <div className="relative flex flex-col items-center justify-center p-8 border-2 border-dashed border-black/20 dark:border-white/20 rounded-xl text-center">
              <span className="material-symbols-outlined text-4xl text-black/40 dark:text-white/40 mb-4">
                upload_file
              </span>
              <h3 className="text-lg font-semibold text-black dark:text-white">
                Drag and drop a CSV file
              </h3>
              <p className="mt-1 text-sm text-black/60 dark:text-white/60">
                The file should contain a column named abstract.
              </p>
              <p className="mt-4 text-sm text-black/60 dark:text-white/60">or</p>
              <button className="mt-4 px-4 py-2 text-sm font-semibold text-white bg-primary/80 hover:bg-primary rounded-lg transition-colors">
                Browse Files
              </button>
              <input
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                type="file"
              />
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
