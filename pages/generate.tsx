"use client"

import { useEffect, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';

export default function RelatedWork() {
    const [idea, setIdea] = useState<string>('…loading');

    const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    useEffect(() => {

        // Get research idea from sessionStorage
        const researchIdea = sessionStorage.getItem('researchIdea');

        if (!researchIdea) {
            setIdea('No research idea provided. Please go back and enter your research idea.');
            return;
        }
        fetch(`${API_URL}/api`,{
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        research_idea: researchIdea
                    }),
                })
            .then(res => res.text())
            .then(setIdea)
            .catch(err => setIdea('Error: ' + err.message));
    },[API_URL]);

    return (
        <div className="flex flex-col min-h-screen">
            {/* Header */}
            <header className="sticky top-0 z-10 bg-background-light/80 dark:bg-background-dark/80 backdrop-blur-sm border-b border-black/10 dark:border-white/10">
                <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center gap-4">
                        <div className="w-8 h-8 text-primary">
                            <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                                <path d="M42.4379 44C42.4379 44 36.0744 33.9038 41.1692 24C46.8624 12.9336 42.2078 4 42.2078 4L7.01134 4C7.01134 4 11.6577 12.932 5.96912 23.9969C0.876273 33.9029 7.27094 44 7.27094 44L42.4379 44Z" fill="currentColor"></path>
                            </svg>
                        </div>
                        <h2 className="text-lg font-bold text-black dark:text-white">ResearchAI</h2>
                    </div>
                    <div className="flex items-center gap-6">
                        <nav className="hidden md:flex items-center gap-6 text-sm font-medium">
                            <a className="text-black/60 hover:text-black dark:text-white/60 dark:hover:text-white transition-colors" href="#">Home</a>
                            <a className="text-black/60 hover:text-black dark:text-white/60 dark:hover:text-white transition-colors" href="#">Projects</a>
                            <a className="text-black/60 hover:text-black dark:text-white/60 dark:hover:text-white transition-colors" href="#">Templates</a>
                            <a className="text-black dark:text-white font-semibold" href="#">Documents</a>
                        </nav>
                        <div className="flex items-center gap-4">
                            <button className="flex items-center justify-center h-10 w-10 rounded-full text-black/60 hover:text-black bg-black/5 dark:bg-white/5 dark:text-white/60 dark:hover:text-white transition-colors">
                                <span className="material-symbols-outlined text-2xl">help</span>
                            </button>
                            <div className="h-10 w-10 rounded-full bg-cover bg-center" style={{backgroundImage: 'url("https://lh3.googleusercontent.com/aida-public/AB6AXuDjMJ1mtNLfwL0QdqL2lFzHZKDuRIaZ1y2NOewJzCa-R28Yu6Ks4WHJhPb78uouLsCbYpTTIMAJOj330sxbz9njZtHHUES8_ZEFe4OUrKDncq7RGffyxklH2EEbBDfHno0jexnZB-F5ei7UFOXA6KSBiwlqeLxx6NOpFRYsWKLUVlL3KWaeFod9GN3HfXDmY2IRESGdJ-lWtxeuu6RDDwvyZufPb9ueXdQOmTIrtLbHdE7dWiTV2DRNHQgRakQONhDg-4Rkf7VibX8")'}}></div>
                        </div>
                    </div>
                </div>
            </header>

            {/* Main Content */}
            <main className="flex-1">
                <div className="mx-auto max-w-4xl py-12 px-4 sm:px-6 lg:px-8">
                    <div className="space-y-8">
                        <div className="flex items-center justify-between">
                            <div>
                                <h1 className="text-4xl font-bold tracking-tight text-black dark:text-white">Related Work</h1>
                                <p className="mt-1 text-sm text-black/50 dark:text-white/50">Generated on {new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' })}</p>
                            </div>
                        </div>
                        <div className="bg-background-light dark:bg-background-dark border border-black/10 dark:border-white/10 rounded-lg shadow-sm">
                            <div className="p-6">
                                <div className="prose prose-lg max-w-none text-black/80 dark:text-white/80">
                                    <ReactMarkdown
                                        remarkPlugins={[remarkGfm, remarkBreaks]}
                                    >
                                        {idea}
                                    </ReactMarkdown>
                                </div>
                            </div>
                            <div className="border-t border-black/10 dark:border-white/10 px-6 py-4 flex justify-end items-center gap-3">
                                <button className="inline-flex items-center justify-center h-10 px-4 rounded-lg text-sm font-semibold bg-primary/10 text-primary hover:bg-primary/20 transition-colors">
                                    Edit
                                </button>
                                <button className="inline-flex items-center justify-center h-10 px-4 rounded-lg text-sm font-semibold bg-primary text-white hover:bg-primary/90 transition-colors">
                                    Export
                                    <span className="material-symbols-outlined text-base ml-1.5 -mr-1">download</span>
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}