import Code from '../../components/code';
import Redirect from '../../components/redirect';
import styles from '../../styles/article.module.scss'

function Article1() {

    return (<div className={styles.article}>

        <h1>A Quick Introduction to the Eigen Library</h1>
        <strong><i>October 5, 2024</i></strong>
        <p>Eigen is a C++ template library for linear algebra, widely used for its performance and convenience. 
            This article overviews key Eigen features, focusing on those used in my <a href='/a/2024-10-06'>blog post</a> about automatic differentiation.
            It's designed for readers new to the library who want a quick start without diving deep into the documentation. 
            For a more comprehensive tutorial on Eigen, please refer to <Redirect href='https://eigen.tuxfamily.org/dox/group__DenseMatrixManipulation__chapter.html'>the official documentation on dense matrices</Redirect>.
        </p>

        <h2>The Matrix Class</h2>
        <p>At the core of Eigen is the <var>Matrix</var> class, defined in the <var>Eigen/Dense</var> header, which represents a dense column-major matrix.</p>
        <Code lang='cpp'>{`Matrix<typename Scalar, int RowsAtCompileTime, int ColsAtCompileTime>`}</Code>
        <p>The template arguments are:</p>
        <ol>
            <li><var>Scalar</var>: The data type of the matrix elements (e.g., float, double, int)</li>
            <li><var>RowsAtCompileTime</var>: The number of rows (use -1 for dynamic size)</li>
            <li><var>ColsAtCompileTime</var>: The number of columns (use -1 for dynamic size)</li>
        </ol>
        <p>For convenience, Eigen provides several type aliases:</p>
        <ul>
            <li><var>{String.raw`MatrixX<typename Scalar>`}</var> : Equivalent to <var>{String.raw`Matrix<typename Scalar, -1, -1>`}</var> (dynamic size)</li>
            <li><var>MatrixXf</var>, <var>MatrixXd</var>, <var>MatrixXi</var> : Dynamic-sized matrices of float, double, and int respectively</li>
        </ul>

        <h2>Basic Usage</h2>
        <p>Let's explore some fundamental operations with Eigen matrices:</p>
        <Code lang='cpp'>{String.raw
`// Fixed-size matrix
Eigen::Matrix<float, 2, 2> a{{1, 2}, {3, 4}};
std::cout << "a is \n" << a << "\n";

// Dynamic-sized matrix
Eigen::Matrix<float, -1, -1> b(4, 3);
b.resize(2, 2);
b.setZero();
b(0, 0) = 2;
b(1, 1) = 2;
std::cout << "b is \n" << b << "\n";

// Type casting
Eigen::MatrixX<double> c = b.cast<double>();

// Matrix multiplication
Eigen::MatrixX<float> d = a * b;
std::cout << "Eigen works: " << std::boolalpha 
            << (d == Eigen::MatrixX<float>{{2, 4}, {6, 8}}) 
            << "\n";

// Complex expression
Eigen::MatrixX<float> e = 
    1.23 * a * b + Eigen::MatrixX<float>::Constant(2, 2, 1.18);
std::cout << "The result is \n" << e << "\n";

// Lazy evaluation
auto f = 2 * (a + b) + 4.56 * (a - b);
Eigen::MatrixX<float> g = f;
std::cout << "f is \n" << g << "\n";

// Reduction methods
std::cout << "f.sum() is " << g.sum() << "\n";
std::cout << "f.prod() is " << g.prod() << "\n";
std::cout << "The mean of f is " << g.mean() << "\n";
std::cout << "The norm of f is " << g.norm() << "\n";

// Element-wise operations
Eigen::MatrixX<float> h = a.array() * b.array();
Eigen::MatrixX<float> i = h.array() + 7.89;

// Coefficient-wise functions
Eigen::MatrixX<float> j = i.cwiseAbs() + i.cwiseInverse();
`}
        </Code>
        <p>This code demonstrates matrix creation, basic operations, lazy evaluation, reduction methods, and element-wise operations.</p>

        <Code title="stdout" lang='text'>{String.raw
`a is 
1 2
3 4
b is 
2 0
0 2
Eigen works: true
The result is 
 3.64   6.1
 8.56 11.02
f is 
 1.44 13.12
19.68 21.12
f.sum() is 55.36
f.prod() is 7852.63
The mean of f is 13.84
The norm of f is 31.7422`}
        </Code>

        <h2>Efficient Function Arguments</h2>
        <p>When passing matrices to functions, it's important to consider efficiency. Here's an example of an inefficient approach:</p>
        <Code lang='cpp'>{String.raw
`void f(const Eigen::MatrixX<float>& m)
{
// some math
}

int main()
{
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(4, 4);
    f(m.topLeftCorner(3,3));
}`}
        </Code>
        <p>In this case, <var>topLeftCorner</var> returns a type different of <var>MatrixXf</var>, causing an implicit copy.</p>
        <p>A better alternative is to use the <var>Eigen::Ref</var> class, which can either represent a <var>MatrixX</var> or a block of it.</p>
        <Code lang='cpp'>{String.raw
`// A function that can modify the matrix passed to it
void f(Eigen::Ref<Eigen::MatrixX<float>> m)
{
    // some math
}

// A function that takes a const Ref 
void g(Eigen::Ref<const Eigen::MatrixX<float>> m)
{
    // some math
}

int main()
{
    Eigen::MatrixXf m = Eigen::MatrixXf::Random(4, 4);
    f(m);
    g(m.topLeftCorner(3,3));
    g(m + Eigen::MatrixXf::Identity(4, 4));
}`}
        </Code>
        <p>The first call to <var>g</var> avoids copying, while the second creates a temporary due to the mismatched memory layout.</p>
        <p>For more detailed information and advanced usage, I highly recommend reading the <Redirect href="https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html">Eigen documentation on this topic</Redirect>.</p>
    </div>)
}

export default Article1;