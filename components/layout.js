import Navbar from './navbar.js'

export default function Layout({ children }) {
    return (
        <>
            <Navbar />
            <div className='content'>
                {children}
            </div>
        </>
    )
}