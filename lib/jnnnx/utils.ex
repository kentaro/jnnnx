defmodule Jnnnx.Utils do
  def to_categorical(t, num) when num >= 0 do
    {shape} = Nx.shape(t)
    o = Nx.tensor(Enum.to_list(0..(num - 1)))

    t
    |> Nx.reshape({shape, 1}, names: [:batch, :output])
    |> Nx.equal(o)
  end
end
