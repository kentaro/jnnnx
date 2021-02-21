defmodule Jnnnx do
  import Nx.Defn

  @default_epoch 5
  @default_batch_size 50
  @default_learning_rate 0.01

  def fit(x, y, opts \\ []) do
    epoch = opts[:epoch] || @default_epoch
    batch_size = opts[:batch_size] || @default_batch_size
    learning_rate = opts[:learning_rate] || @default_learning_rate
    opts = Keyword.put(opts, :learing_rate, learning_rate)

    zip =
      Enum.zip(
        Nx.to_batched_list(x, batch_size),
        Nx.to_batched_list(y, batch_size)
      )
      |> Enum.with_index()

    for e <- 1..epoch,
        {{x_batch, y_batch}, b} <- zip,
        reduce: Jnnnx.MNIST.init_params() do
      params ->
        IO.puts "epoch #{e}, batch #{b}"
        Jnnnx.update(params, x_batch, y_batch, opts)
    end
  end

  defn evaluate({w1, b1, w2, b2}, x, y) do
    x_result =
      predict({w1, b1, w2, b2}, x)
      |> Nx.argmax(axis: :output)
    y_result =
      y |> Nx.argmax(axis: :output)
    count =
      Nx.equal(x_result, y_result)
      |> Nx.sum()
    {total} = Nx.shape(x_result)
    Nx.divide(count, total)
  end

  defn predict({w1, b1, w2, b2}, batch) do
    batch
    |> Nx.dot(w1)
    |> Nx.add(b1)
    |> Nx.logistic()
    |> Nx.dot(w2)
    |> Nx.add(b2)
    |> softmax()
  end

  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t), axes: [:output], keep_axes: true)
  end

  defn loss({w1, b1, w2, b2}, x, y) do
    preds = predict({w1, b1, w2, b2}, x)
    -Nx.sum(Nx.mean(Nx.log(preds) * y, axes: [:output]))
  end

  defn update({w1, b1, w2, b2} = params, x, y, opts \\ []) do
    {grad_w1, grad_b1, grad_w2, grad_b2} = grad(params, loss(params, x, y))

    {
      w1 - grad_w1 * opts[:learning_rate],
      b1 - grad_b1 * opts[:learning_rate],
      w2 - grad_w2 * opts[:learning_rate],
      b2 - grad_b2 * opts[:learning_rate]
    }
  end
end
