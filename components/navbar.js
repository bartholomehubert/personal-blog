import styles from './navbar.module.scss';

export default function Navbar() {

    return (
        <nav className={styles.navbar}>
            <div className={styles['navbar-left']}>
                <a href="/">B. Hubert's blog</a>
            </div>
            <div className={styles['navbar-right']}>
                <a href="/">Blog</a>
                <a href="/about">About</a>
            </div>
        </nav>
    );
}

