# Jnnnx

This is an example implementation of a neural network with [Nx](https://github.com/elixir-nx/nx) presented in José Valim's talk, [Introducing Nx](https://www.youtube.com/watch?v=fPKMmJpAGWc), at Lambda Days 2021.

What I did for this repository were making some arrangement for ease to use and adding some util functions.

## Usage

You can try Nx in a Keras-like manner as below:

```elixir
# loading data
[x_train, y_train, x_test, y_test] = Jnnnx.MNIST.Dataset.load_data()

# normalization
x_train =
  x_train
  |> Nx.reshape({60000, 28*28}, names: [:batch, :input])
  |> Nx.divide(255)
x_test =
  x_test
  |> Nx.reshape({10000, 28*28}, names: [:batch, :input])
  |> Nx.divide(255)

# one-hot-encoding
y_train = y_train |> Jnnnx.Utils.to_categorical(10, names: [:batch, :output])
y_test  = y_test  |> Jnnnx.Utils.to_categorical(10, names: [:batch, :output])

# training
params = Jnnnx.fit(x_train, y_train, epoch: 5, batch_size: 50, learning_rate: 0.01)

# evaluation
score = Jnnnx.evaluate(params, x_test, y_test)
IO.puts("Accuracy: #{Nx.to_scalar(score)}")
```

The above code is copied to `example/keras_like_manner.exs`.  
You can also try it as the follow.

```shell
$ mix run example/keras_like_manner.exs
```

## Author

Kentaro Kuribayashi &lt;kentarok@gmail.com&gt;

This repository has a bunch of manually-copied codes from the original author's talk, [Introducing Nx - José Valim | Lambda Days 2021](https://www.youtube.com/watch?v=fPKMmJpAGWc).
