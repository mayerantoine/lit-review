"use client"

import { useEffect, useState } from 'react';
import { useRouter } from 'next/router';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import remarkBreaks from 'remark-breaks';
import { fetchEventSource, EventSourceMessage } from '@microsoft/fetch-event-source';

interface Citation {
    id: number;
    title: string;
    abstract: string;
}

export default function RelatedWork() {
    const router = useRouter();
    const [idea, setIdea] = useState<string>('');
    const [isLoading, setIsLoading] = useState<boolean>(true);
    const [error, setError] = useState<string>('');
    const [citations, setCitations] = useState<Citation[]>([]);

    //const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

    useEffect(() => {
        // Get research idea from sessionStorage
        const researchIdea = sessionStorage.getItem('researchIdea');

        if (!researchIdea) {
            setIdea('No research idea provided. Please go back and enter your research idea.');
            setIsLoading(false);
            return;
        }

        // Use fetchEventSource for SSE streaming
        const controller = new AbortController();

        fetchEventSource('/api/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                research_idea: researchIdea
            }),
            signal: controller.signal,

            async onopen(response) {
                if (response.ok) {
                    setIsLoading(false);
                    return; // Success
                } else if (response.status >= 400 && response.status < 500 && response.status !== 429) {
                    // Client error - don't retry
                    const errorData = await response.json();
                    throw new Error(errorData.detail || 'Failed to generate review');
                } else {
                    // Server error or rate limit - could retry
                    throw new Error('Server error occurred');
                }
            },

            onmessage(event: EventSourceMessage) {
                const data = event.data;

                // Check for special messages
                if (data.startsWith('[METADATA]')) {
                    // Handle metadata and extract citations
                    try {
                        const metadata = JSON.parse(data.substring(10));
                        console.log('Generation metadata:', metadata);
                        if (metadata.references && Array.isArray(metadata.references)) {
                            setCitations(metadata.references);
                        }
                    } catch (_e) {
                        console.error('Failed to parse metadata:', _e);
                    }
                } else if (data === '[DONE]') {
                    // Stream complete
                    setIsLoading(false);
                } else if (data.startsWith('[ERROR]')) {
                    // Error occurred
                    try {
                        const errorData = JSON.parse(data.substring(7));
                        setError(errorData.message || 'An error occurred during generation');
                    } catch {
                        setError('An error occurred during generation');
                    }
                    setIsLoading(false);
                } else {
                    // Regular text chunk - append to idea
                    setIdea(prev => prev + data);
                }
            },

            onerror(err: unknown) {
                setIsLoading(false);
                const errorMessage = err instanceof Error ? err.message : 'Connection error. Please try again.';
                setError(errorMessage);
                throw err; // Stop retrying
            },

            onclose() {
                // Stream closed
                setIsLoading(false);
            }
        }).catch((err: unknown) => {
            setIsLoading(false);
            if (err instanceof Error && err.name !== 'AbortError') {
                setError(err.message || 'Failed to generate review');
            }
        });

        // Cleanup function to abort the request if component unmounts
        return () => {
            controller.abort();
        };
    }, []);

    const downloadAsMarkdown = () => {
        // Build markdown content
        const currentDate = new Date().toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });

        let markdown = `# Related Work\n\n`;
        markdown += `Generated on ${currentDate}\n\n`;
        markdown += `${idea}\n\n`;

        // Add references section if citations exist
        if (citations.length > 0) {
            markdown += `## References\n\n`;
            citations.forEach((citation) => {
                markdown += `- [${citation.id}] **${citation.title}**\n`;
                markdown += `  ${citation.abstract}\n\n`;
            });
        }

        // Create blob and download
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `related-work-${new Date().toISOString().split('T')[0]}.md`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    };

    return (
        <div className="flex flex-col min-h-screen">
            {/* Header */}
            <header className="sticky top-0 z-10 bg-background-light/80 dark:bg-background-dark/80 backdrop-blur-sm border-b border-black/10 dark:border-white/10">
                <div className="mx-auto flex h-16 max-w-7xl items-center justify-between px-4 sm:px-6 lg:px-8">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 text-primary">
                            <svg fill="none" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
                                <path d="M42.4379 44C42.4379 44 36.0744 33.9038 41.1692 24C46.8624 12.9336 42.2078 4 42.2078 4L7.01134 4C7.01134 4 11.6577 12.932 5.96912 23.9969C0.876273 33.9029 7.27094 44 7.27094 44L42.4379 44Z" fill="currentColor"></path>
                            </svg>
                        </div>
                        <h1 className="text-lg font-bold text-black dark:text-white">ResearchAI</h1>
                    </div>
                    <button
                        onClick={() => router.push('/')}
                        className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-primary hover:bg-primary/10 rounded-lg transition-colors"
                    >
                        <span className="material-symbols-outlined">add</span>
                        New Search
                    </button>
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
                                {error && (
                                    <div className="mb-6 p-4 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
                                        <div className="flex items-start gap-3">
                                            <span className="material-symbols-outlined text-red-600 dark:text-red-400 flex-shrink-0">error</span>
                                            <div>
                                                <h3 className="text-sm font-semibold text-red-800 dark:text-red-200 mb-1">Generation Error</h3>
                                                <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {isLoading && !error && !idea && (
                                    <div className="flex items-center justify-center gap-3 p-8">
                                        <div className="w-5 h-5 border-2 border-primary border-t-transparent rounded-full animate-spin"></div>
                                        <span className="text-sm font-medium text-primary">Generating related work section...</span>
                                    </div>
                                )}

                                {!error && idea && (
                                    <>
                                        <div className="prose prose-lg max-w-none text-black/80 dark:text-white/80">
                                            <ReactMarkdown
                                                remarkPlugins={[remarkGfm, remarkBreaks]}
                                            >
                                                {idea}
                                            </ReactMarkdown>
                                            {isLoading && (
                                                <span className="inline-block w-2 h-4 bg-primary animate-pulse ml-1"></span>
                                            )}
                                        </div>

                                        {citations.length > 0 && (
                                            <div className="mt-8 pt-6 border-t border-black/10 dark:border-white/10">
                                                <h3 className="text-xl font-bold text-black dark:text-white mb-4">References</h3>
                                                <div className="space-y-4">
                                                    {citations.map((citation) => (
                                                        <div key={citation.id} className="text-sm">
                                                            <div className="flex gap-2">
                                                                <span className="font-mono text-primary flex-shrink-0">[{citation.id}]</span>
                                                                <div>
                                                                    <p className="font-semibold text-black dark:text-white mb-1">
                                                                        {citation.title}
                                                                    </p>
                                                                    <p className="text-black/60 dark:text-white/60 text-xs line-clamp-3">
                                                                        {citation.abstract}
                                                                    </p>
                                                                </div>
                                                            </div>
                                                        </div>
                                                    ))}
                                                </div>
                                            </div>
                                        )}
                                    </>
                                )}
                            </div>
                            <div className="border-t border-black/10 dark:border-white/10 px-6 py-4 flex justify-end items-center gap-3">
                                <button
                                    onClick={downloadAsMarkdown}
                                    disabled={isLoading || !idea}
                                    style={{ backgroundColor: '#1173d4' }}
                                    className="inline-flex items-center justify-center h-10 px-4 rounded-lg text-sm font-semibold text-white hover:opacity-90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                                >
                                    Download
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