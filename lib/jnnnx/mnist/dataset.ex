defmodule Jnnnx.Mnist.Dataset do
  @doc """
  Load MNIST datasets.

  ## Examples

      iex> [x_train, y_train, x_test, y_test] = Jnnnx.Mnist.Dataset.load_data()

  """
  def load_data() do
    [x_train, y_train] =
      [
        'train-images-idx3-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
      ]
      |> Enum.map(fn f ->
        f
        |> load_data_for()
        |> load_images()
      end)

    [x_test, y_test] =
      [
        'train-labels-idx1-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
      ]
      |> Enum.map(fn f ->
        f
        |> load_data_for()
        |> load_labels()
      end)

    [x_train, y_train, x_test, y_test]
  end

  defp load_data_for(file_name) do
    tmp_file = Path.absname(file_name, "tmp")

    if File.exists?(tmp_file) do
      tmp_file |> File.read!()
    else
      data = file_name |> url_for() |> download_for()
      File.write!(tmp_file, data)
      data
    end
    |> :zlib.gunzip()
  end

  defp url_for(file_name) do
    "http://yann.lecun.com/exdb/mnist/#{file_name}"
  end

  defp download_for(url) do
    res = HTTPoison.get!(url)
    res.body
  end

  defp load_images(data) do
    <<_::32, n_images::32, n_rows::32, n_cols::32, images::binary>> = data

    Nx.from_binary(images, {:u, 8})
    |> Nx.reshape({n_images, n_rows * n_cols})
  end

  defp load_labels(data) do
    <<_::32, _::32, labels::binary>> = data

    Nx.from_binary(labels, {:u, 8})
  end
end
