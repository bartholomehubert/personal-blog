import styles from'../styles/about.module.scss';

function About() {
    return (
        <div className={styles.about}>
            <h1>About</h1>
            <div className={styles.author}>
                <p>My name is Bartholomé Hubert. On this blog I share my passion for learning about computers and math, which I do during my free time. If you find mistakes, imprecisions or just want to make a suggestion, do not hesistate to open a Github issue or contact me on my email address.</p>
            </div>

            <div className={styles.contact}>
            <div className={styles.links}>
                <a href='https://github.com/bartholomehubert' target="_blank" rel="noopener noreferrer">
                    <img src="icon-github.svg" alt="github icon" />
                    <p>github.com</p>
                </a>
                <a href='https://substack.com/@bartholomehubert' target="_blank" rel="noopener noreferrer">
                    <img src='icon-substack.png' alt='substack icon' />
                    <p>substack.com</p>
                </a>
            </div>
            <div className={styles.email}>
                <h3>Email</h3>
                <p>bartholomehubert@outlook.com</p>
            </div>
            </div>
        </div>
    );
}

export default About;
