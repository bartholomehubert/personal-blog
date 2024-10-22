import { Html, Head, Main, NextScript } from 'next/document'
import Navbar from '../components/navbar'

export default function Document() {
    return (
        <Html lang="en">
            <Head>
                <meta name="theme-color" content="#000" />
                <meta name="B. Hubert's blog" />
                <meta name="description" content="A personnal blog about " />
                <link rel="icon" href="/favicon.png" />
                <link rel="apple-touch-icon" href="/logo192.png" />
                <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans" />
                <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.10.0/styles/atom-one-dark.min.css" />            </Head>
            <body>
                <Main />
                <NextScript />
            </body>
        </Html>
    )
}