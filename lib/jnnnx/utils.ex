defmodule Jnnnx.Utils do
  def to_categorical(t, num, opts \\ []) do
    size = Nx.size(t)
    o = Nx.tensor(Enum.to_list(0..(num - 1)))

    t
    |> Nx.reshape({size, 1}, opts)
    |> Nx.equal(o)
  end
end
