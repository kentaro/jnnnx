defmodule Jnnnx.Utils do
  import Nx.Defn

  @doc """
  Converts a class vector to a matrix of one-hot-encoded vectors.

  ## Examples

  ```
  iex> Jnnnx.Utils.to_categorical(y_train, 10)
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

  @doc """
  Implements the softmax function.

  ## Examples

  ```
  iex> t = Nx.tensor([1,2,3,4])
  #Nx.Tensor<
    s64[4]
    [1, 2, 3, 4]
  >
  iex> Jnnnx.Utils.softmax(t)
  #Nx.Tensor<
    f64[4]
    [0.03205860328008499, 0.08714431874203257, 0.23688281808991013, 0.6439142598879722]
  >
  ```

  """
  defn softmax(t) do
    Nx.exp(t) / Nx.sum(Nx.exp(t))
  end
end
