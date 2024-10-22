import { useRef, useEffect } from "react";
import hljs from "highlight.js";

export default function Code(prop) {
    const codeRef = useRef(null);

    useEffect(() => {
        hljs.configure({ useBR: true });
        if (!codeRef.current.getAttribute('data-highlighted')) {
            hljs.highlightElement(codeRef.current);
        }
    }, [prop.children]);

    return (
        <pre title={prop.title}>
            <code ref={codeRef} className={`language-${prop.lang}`}>
                {prop.children}
            </code>
        </pre>
    );
};
