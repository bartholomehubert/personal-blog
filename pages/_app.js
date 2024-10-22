import Head from 'next/head';
import Layout from '../components/layout';
import '../styles/global.scss';
import MathJax from '../components/mathjax';

export default function App({ Component, pageProps }) {
    return (
        <>
            <Head>
                <title>B. Hubert's blog</title>
            </Head>
            <MathJax>
                <Layout>
                    <Component {...pageProps} />
                </Layout>
            </MathJax>
        </>
    )
}