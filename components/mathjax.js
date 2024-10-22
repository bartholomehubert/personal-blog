'use client'
import { useEffect } from 'react';

export default function MathJax({ children }) {
  useEffect(() => {
    // Add MathJax script and configuration
    const script = document.createElement('script');
    script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
    script.async = true;

    // Add script to document
    document.head.appendChild(script);

    // Cleanup
    return () => {
      document.head.removeChild(script);
    };
  }, []);

  return <div>{children}</div>;
}