# Quantization

Quantization stores a model's weights in fewer bits. It cuts the memory a model needs, and often speeds it up, at the cost of some quality.

| Level | Web UI | CLI | Memory |
|---|---|---|---|
| None | **Quant: None** | | Full size |
| 8-bit | **Quant: q8** | `-q 8` | About half |
| 4-bit | **Quant: q4** | `-q 4` | About a quarter |

=== "macOS"

    Both levels use mflux's quantization. On a 16 GB Mac, Z-Image Turbo and FLUX.2 Klein need `q8` or `q4`.

=== "Windows and Linux"

    `q8` is FP8 and `q4` is NF4.

    - **Unquantized and `q8`** keep the model in system memory and move it to the GPU piece by piece, so they need only about 4 GB of GPU memory. `q8` halves the system memory a model needs, which makes large models much faster on machines that can't hold them unquantized.
    - **`q4`** keeps the whole model on the GPU, so the GPU must hold it at 4 bits (Krea 2: about 7 GB).

The Web UI's [memory badge](memory.md#the-memory-badges) follows the level you pick, so you can see which one fits.

Video uses the same idea through its two models: `ltx-4` and `ltx-8` are LTX-2.3 at 4 and 8 bits.

## Stored quants

The first time a model runs at `q8` or `q4`, its quantized weights are saved in `~/.ziv/models/` as `<name>@q8` or `<name>@q4`, for example `zit@q8`. Later jobs load that copy instead of quantizing again, so they start faster. On Windows and Linux they also never briefly need the unquantized model's memory.

The first job shows *Saving a q8 copy…* and takes longer: about 20–30 seconds on macOS, a few minutes on Windows and Linux. The copy needs roughly half (`q8`) or a quarter (`q4`) of the model's disk space.

- The copy holds the base model only. LoRAs are applied on top when a job loads.
- It is replaced automatically when the model's weights change, or when an update changes how copies are made. The next job then rebuilds it once.
- The Models page lists it as *q8 copy of &lt;name&gt;*, where you can delete it. It doesn't appear in the model picker: choose the model and a quant level.
- Deleting a model also deletes its copies.
- A model given as a raw Hugging Face repo or a folder outside `~/.ziv/models/` is quantized at load every time. Use an alias, or install the model, to get a copy.

You can also save a copy when you convert a checkpoint (`--quantize 8`, or **Quantized copy** on the Models page).
