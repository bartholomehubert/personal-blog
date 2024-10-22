
export default function Custom404(props) {
    return (
        <>
        <h1 style={{'textAlign':'center', 'marginTop':'20vh'}}>404</h1>
        <h1 style={{'textAlign':'center', 'marginBottom':'20vh'}}>Page not found</h1>
        {
            (props.details) ? (<h3 style={{'textAlign':'center'}}>Details: {props.details}</h3>) : null
        }
        </>
    );
}