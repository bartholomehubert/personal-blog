import Image from 'next/image';
import styles from '../styles/home.module.scss';

function Home() {
    return (
        <div className={styles.homepage}>
            <h1>Articles</h1>
            <div className={styles['articles-thumbnail']}>

                <div className={styles['article-thumbnail']}>
                    <a href='/a/2024-10-05'>
                        <h2>A Quick Introduction to the Eigen Library</h2>
                    </a>
                </div>

                <div className={styles['article-thumbnail']}>
                    <a href='/a/2024-10-06'>
                        <h2>Automatic Differentiation in C++ from Scratch</h2>
                        <img  src='/a2-3.png' alt='computational graph illustration' />
                    </a>
                </div>

                <div className={styles['article-thumbnail']}>
                    <a href='/a/2024-10-09'>
                        <h2>Training a Neural Network with Automatic Differentiation in C++</h2>
                        <video width="500" autoPlay loop muted>
                            <source src="a3-anim.mp4" type="video/mp4" />
                        </video>
                    </a>
                </div>


                {/* <div className={styles['article-thumbnail']}>
                    <a href='/a/todo1'>
                        <h2>Creating Arbitrary Precision Floats in C++</h2>
                    </a>
                </div> */}


                <div className={styles['article-thumbnail']}>
                    <a href='/a/todo2'>
                        <h2>An Experiment on Forecasting Crypto Price Returns with LSTMs</h2>
                    </a>
                </div>


                <div className={styles['article-thumbnail']}>
                    <a href='/a/todo3'>
                        <h2>My Linux Scripts and Dotfiles (On Debian with i3)</h2>
                    </a>
                </div>



            </div>
        </div>
    );
}

export default Home;
