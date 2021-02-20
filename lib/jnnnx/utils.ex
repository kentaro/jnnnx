defmodule Jnnnx.Utils do
  @doc """
  Converts a class vector to a matrix of one-hot-encoded vectors.

  ## Examples

  ```
  iex(2)> y_train = Jnnnx.Utils.to_categorical(y_train, 10)
  #Nx.Tensor<
    u8[60000][10]
    [
      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
      ...
    ]
  >
  ```

  """
  def to_categorical(t, num) when num >= 0 do
    {shape} = Nx.shape(t)
    o = Nx.tensor(Enum.to_list(0..(num - 1)))

    t
    |> Nx.reshape({shape, 1})
    |> Nx.equal(o)
  end
end
