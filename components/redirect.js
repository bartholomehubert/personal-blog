
export default function Redirect(prop) {
    return <a href={prop.href} target="_blank" rel="noopener noreferrer">{prop.children}</a> 
}