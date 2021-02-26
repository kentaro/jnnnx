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
