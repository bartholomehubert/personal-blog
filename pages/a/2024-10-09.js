import { useEffect } from 'react';
import Code from '../../components/code';
import Redirect from '../../components/redirect';

import styles from '../../styles/article.module.scss'

function Article3() {

    return (<div className={styles.article}>
        <h1>Training a Neural Network with Automatic Differentiation in C++</h1>
        <strong><i>October 9, 2024</i></strong>
        <p>In the <Redirect href='/a/2024-10-06'>previous post</Redirect>, we implemented automatic differentiation with computational graphs.
            Let's now implement a neural network that will learn to memorize a mathematical function.
        </p>
        <h2>Prerequisites</h2>
        <p>This is a follow-up to my previous post on automatic differentiation, you should have read it in order to understand the implementation. This post also assumes you are familiar with neural networks and C++. </p>
        <h2>The implementation</h2>
        <h3>Layers</h3>
        <p>The <var>Layer</var> class will be a pure virtual interface that defines two methods:</p>
        <ul>
            <li><var>forward</var>: will compute the result</li>
            <li><var>update_weights</var>: will update the weights after the gradient has been calculated</li>
        </ul>
        <Code title='layer.hpp' lang='cpp'>
            {String.raw
                `#pragma once
#include "matrix.hpp"
#include <functional>
#include <random>

template<typename T>
using Activation = std::function<Matrix<T>(const Matrix<T>&)>;

template<typename T>
class Layer
{
public:
    Layer() = default;
    Layer(const Activation<T>& a) : activation(a) {}

    virtual Matrix<T> forward(const Matrix<T>& input) = 0;
    virtual void update_weights(T lr) = 0;
    Activation<T> activation;
};       
`}
        </Code>
        <p>The <var>Dense</var> class implements a fully connected layer that inherits the <var>Layer</var> interface.</p>
        <Code title='layer.hpp' lang='cpp'>
            {String.raw
                `/* ... */
template<typename T>
class Dense: public Layer<T>
{
public:
    Dense(size_t n_inputs, size_t n_outputs, const Activation<T>& a = [] (const Matrix<T>& z) { return z; }) 
        : Layer<T>(a) 
    {
        // uniform Glorot initialization
        const double r = sqrt(6. / (n_inputs + n_outputs));

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(-r, r);
        
        W = Matrix<double>(Eigen::MatrixX<T>::NullaryExpr(n_inputs, n_outputs, [&](){return dis(gen);}));
        b = Matrix<double>(Eigen::MatrixX<T>::Constant(1, n_outputs, 0));

        W.set_requires_gradient();
        b.set_requires_gradient();
    }

    Matrix<T> forward(const Matrix<T>& input) override 
    {
        return this->activation(input * W + b);
    }

    void update_weights(T lr) override
    {
        W = Matrix<double>(W.eigen() - lr * W.gradient());
        b = Matrix<double>(b.eigen() - lr * b.gradient());
        W.set_requires_gradient();
        b.set_requires_gradient();
    }

    Matrix<T> W;
    Matrix<T> b;
};
`}
        </Code>
        <p>The weights are initialized using the Glorot initialization technique.
            We need to call the <var>set_requires_gradient</var> method in order to accumulate the gradient in the backward phase.
        </p>
        <p>In the <var>update_weights</var> method, we assign a new matrix to each weight.
            While the weight update implementation is simple, it could be optimized to reduce unnecessary creations and deletion of shared pointers.
        </p>

        <h3>The Network</h3>
        <p>To manage our layers, we implement a <var>Network</var> class that will store a vector of unique pointers to its layers.</p>
        <Code title='network.hpp' lang='cpp'>
            {String.raw
                `#pragma once
#include <vector>
#include <memory>
#include "layer.hpp"

template<typename T>
class Network
{
public:
    void add_layer(std::unique_ptr<Layer<T>>&& l) 
    {
        layers.push_back(std::move(l));
    }

    Matrix<T> forward(const Matrix<T>& input)
    {
        Matrix<T> output = input;
        for (const auto& l : layers) {
            output = l->forward(output);
        }
        return output;
    }

    std::vector<Matrix<T>> predict(const std::vector<Matrix<T>>& inputs)
    {
        std::vector<Matrix<T>> outputs;
        for (const Matrix<T>& input : inputs) {
            outputs.push_back(forward(input));
        }
        return outputs;
    }

    void update_weights(T lr) const
    {
        for (const auto& l : layers) {
            l->update_weights(lr);
        }
    }

    std::vector<std::unique_ptr<Layer<T>>> layers;
};
`}
        </Code>
        <p>The <var>forward</var> method computes the result.
            Note that matrix assignments only copy shared pointers, not the underlying data.</p>

        <h2>Training the network</h2>

        <h3>Dataset Generation</h3>
        <p>First, let's generate our training data using Python:</p>
        <Code title='generate.ipynb cell 1' lang='py'>
            {String.raw
                `import matplotlib.pyplot as plt
import numpy as np
plt.style.use('dark_background')

l = np.linspace(-1, 1, 50)
real_grid, imag_grid = np.meshgrid(l, l)
complex_grid = real_grid + imag_grid * 1j

r = np.abs(complex_grid)
theta = np.angle(complex_grid)

z = np.sin(4 * (r + theta))

# saves the figure to a flat sequence of float64
z.tofile('./figure.bin')

# changes the appearance of the figure
plt.figure()
plt.imshow(z, cmap='magma')
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()`}
        </Code>
        <img src='/a3-1.png' alt='the function representation' />
        <p>This code generates a whirlpool-like function that will be challenging for the network to learn.
            Let's break down the steps:
        </p>
        <ol>
            <li>Creating a grid of complex number grid spanning from \(-1\) to \(1\) on the real axis and from \(-i\) to \(i\) on the imaginary axis.</li>
            <li>Extracting the norm and the argument of each coordinate in our grid.</li>
            <li>Computing \(\sin(4(\theta+r))\)</li>
        </ol>
        <p>You can think of the third operation as the height of a sine wave that depends only on \(theta\) around a circle of constant radius \(r\).
            The sine wave is then shifted as \(r\) increases.
            We multiply by \(4\) to create a pattern of four minima and maxima on circles of different values of \(r\).
        </p>

        <h3>The training loop</h3>
        <p>We created a file called 'figure.bin' in the 'data' subfolder.
            The following program parses this file:</p>
        <Code title='main.cpp' lang='cpp'>
            {String.raw
                `#include <fstream>
#include <iostream>
#include <vector>
#include <filesystem>

#include "matrix.hpp"
#include "network.hpp"

// linearly spaced values in the range -1 to 1 with n elements
double linspace(size_t i, size_t n)
{
    return -1. + (2. / (n - 1)) * i;
}

int main()
{
    size_t nbytes = std::filesystem::file_size(std::filesystem::relative("../data/figure.bin"));
    size_t n = sqrt(nbytes / sizeof(double));

    if (nbytes % sizeof(double) != 0 || n * n * sizeof(double) != nbytes) {
        throw std::runtime_error{ "Invalid figure file" };
    }

    std::ifstream figure_stream("../data/figure.bin", std::ios_base::binary);
    std::ofstream predictions_stream("../data/predictions.bin", std::ios_base::binary);
    std::vector<Matrix<double>> x;
    std::vector<Matrix<double>> y;

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double val;
            figure_stream.read(reinterpret_cast<char*>(&val), sizeof(double));
            x.push_back(Matrix<double>{{linspace(i, n), linspace(j, n)}});
            y.push_back(Matrix<double>{{val}});
        }
    }
}`}
        </Code>
        <p>
            The <var>predictions_stream</var> will receive the predictions of our model during its training.
            The <var>x</var> vector represents the grid we used in Python, but this time not with complex numbers but with row vectors as the coordinates.
        </p>
        <Code title='main.cpp' lang='cpp'>
            {String.raw
                `int main()
{
    /* ... */
    Network<double> model;
    const Activation<double> relu = [](const Matrix<double>& z) { return z.cwise_max(); };
    model.add_layer(std::make_unique<Dense<double>>(2, 16, relu));
    model.add_layer(std::make_unique<Dense<double>>(16, 16, relu));
    model.add_layer(std::make_unique<Dense<double>>(16, 16, relu));
    model.add_layer(std::make_unique<Dense<double>>(16, 16, relu));
    model.add_layer(std::make_unique<Dense<double>>(16, 1));

    const double lr = 0.001;
    const size_t epochs = 1000;
    const size_t n_samples = x.size();
}`}
        </Code>
        <p>We construct the model with its settings.</p>
        <Code title='main.cpp' lang='cpp'>
            {String.raw
                `int main()
{
    /* ... */
    for (size_t e = 0; e < epochs; ++e) {
        double loss_acc = 0;
        for (size_t i = 0; i < n_samples; ++i) {
            Matrix<double> output = model.forward(x[i]);
            Matrix<double> loss = (output - y[i]).norm();
            loss.backward();
            model.update_weights(lr);
            loss_acc += loss(0, 0);
            std::cout << "\rEpoch: " << e << " Loss: " << loss_acc / (i + 1);

        }
        std::vector<Matrix<double>> prediction = model.predict(x);
        for (const Matrix<double>& pred : prediction) {
            predictions_stream.write(reinterpret_cast<const char*>(&pred(0, 0)), sizeof(double));
        }
        std::cout << '\n';
    }
}`}
        </Code>
        <p>We use <var>loss_acc</var> to accumulate the value of the loss, its purpose is just for debugging.</p>
        <p>At the end of every epoch, we send the predictions of our model to the <var>predictions_stream</var>, which will be used later to create a visualization.</p>

        <h3>Visualizing the training process</h3>
        <p>To visualize our model's learning progress, we can create an animation using this Python script:</p>
        <Code title='generate.ipynb cell 2' lang='py'>
            {String.raw
                `import matplotlib.animation as animation

fig = plt.figure()

# Create axes
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)

# Load data
preds = np.fromfile('./predictions.bin').reshape(-1, 50, 50)


frames = [[ax1.imshow(pred, animated=True, cmap='magma', aspect='equal')] for pred in preds]

for ax in [ax1, ax2]:
    ax.set_xticks([])
    ax.set_yticks([])

# An animation of 4000ms
ani = animation.ArtistAnimation(fig, frames, interval=4000 / preds.shape[0])

ax2.imshow(z, cmap='magma', aspect='equal')

ani.save("./visualization.mp4", dpi=100, writer='ffmpeg')
plt.close()
                `}
        </Code>
        <video width="500" autoPlay loop muted>
            <source src="/a3-anim.mp4" type="video/mp4" />
        </video>
        <p>This visualization shows the network's predictions evolving alongside the target function, providing a visual insight into the learning process.</p>

        <h2>Conclusion</h2>
        <p>We've demonstrated how to combine automatic differentiation with neural networks to create a network capable of learning patterns.
            The resulting implementation, while straightforward, shows the power of automatic differentiation in building neural networks from scratch.
        </p>

    </div>
    );
}

export default Article3;
