defmodule Jnnnx.Utils do
  import Nx.Defn

  def to_categorical(t, num) do
    {shape} = Nx.shape(t)
    o = Nx.tensor(Enum.to_list(0..(num - 1)))

    t
    |> Nx.reshape({shape, 1}, names: [:batch, :output])
    |> Nx.equal(o)
  end
end
